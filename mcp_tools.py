# mcp_tools.py
import pandas as pd
import numpy as np
from datetime import datetime
import json


class MCPServer:
    """MCP Server - Secure tool layer that provides filtered data summaries only"""

    def __init__(self, csv_path="product_data.csv"):
        self.csv_path = csv_path
        self.df = None
        self._load_data()

    def _load_data(self):
        """Private method to load and prepare data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            # Compute discount percentage
            self.df['discount_percentage'] = np.where(
                (self.df['compare_at_price'] > 0) & (self.df['price'] > 0),
                ((self.df['compare_at_price'] - self.df['price']) / self.df['compare_at_price'] * 100),
                0
            )
            # Fill NaN values
            self.df.fillna({
                "title": "Produit sans titre",
                "vendor": "Inconnu",
                "product_type": "Non défini",
                "price": 0,
                "available": 0,
                "discount_percentage": 0
            }, inplace=True)
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            self.df = pd.DataFrame()

    def get_kpi_summary(self):
        """Returns key performance indicators - safe to share with LLM"""
        if self.df.empty:
            return "Aucune donnée disponible"

        total_products = len(self.df)
        avg_price = self.df['price'].mean()
        total_stock = self.df['available'].sum()
        low_stock_count = len(self.df[self.df['available'] < 5])
        high_discount_count = len(self.df[self.df['discount_percentage'] > 20])

        return {
            "total_products": total_products,
            "average_price": round(avg_price, 2),
            "total_stock": int(total_stock),
            "low_stock_products": low_stock_count,
            "high_discount_products": high_discount_count,
            "unique_vendors": self.df['vendor'].nunique(),
            "unique_categories": self.df['product_type'].nunique()
        }

    def get_critical_products(self, stock_threshold=2, discount_threshold=20):
        """Returns critical products with low stock and high discount"""
        if self.df.empty:
            return []

        critical = self.df[
            (self.df['available'] < stock_threshold) &
            (self.df['discount_percentage'] > discount_threshold)
            ]

        # Return only essential info, not full data
        return critical[['title', 'price', 'available', 'discount_percentage', 'vendor']].head(10).to_dict('records')

    def get_vendor_summary(self):
        """Returns vendor performance summary"""
        if self.df.empty:
            return []

        vendor_stats = self.df.groupby('vendor').agg({
            'title': 'count',
            'price': 'mean',
            'available': 'sum',
            'discount_percentage': 'mean'
        }).round(2)

        vendor_stats.columns = ['total_products', 'avg_price', 'total_stock', 'avg_discount']
        return vendor_stats.head(10).to_dict('index')

    def get_category_summary(self):
        """Returns category performance summary"""
        if self.df.empty:
            return []

        category_stats = self.df.groupby('product_type').agg({
            'title': 'count',
            'price': 'mean',
            'available': 'sum'
        }).round(2)

        category_stats.columns = ['total_products', 'avg_price', 'total_stock']
        return category_stats.head(10).to_dict('index')


class MCPClient:
    """MCP Client - LLM wrapper that prepares structured prompts"""

    def __init__(self, groq_api_key):
        from groq import Groq
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"
        self.server = MCPServer()

    def _create_context(self, user_query):
        """Create structured context for the LLM"""
        kpi = self.server.get_kpi_summary()

        context = f"""
CONTEXTE ECOMMERCE:
- Total produits: {kpi['total_products']}
- Prix moyen: {kpi['average_price']}$
- Stock total: {kpi['total_stock']} unités
- Produits en rupture (<5): {kpi['low_stock_products']}
- Produits en forte remise (>20%): {kpi['high_discount_products']}
- Vendeurs uniques: {kpi['unique_vendors']}
- Catégories uniques: {kpi['unique_categories']}

INSTRUCTIONS:
- Répondez en français
- Basez-vous uniquement sur ces données
- Soyez précis et professionnel
- Si des détails spécifiques sont demandés, mentionnez que vous pouvez fournir des analyses supplémentaires
"""
        return context

    def _handle_specific_queries(self, user_query):
        """Handle specific queries that need detailed data"""
        query_lower = user_query.lower()

        if "critique" in query_lower and "stock" in query_lower:
            critical_products = self.server.get_critical_products()
            return f"PRODUITS CRITIQUES:\n{json.dumps(critical_products, indent=2, ensure_ascii=False)}"

        elif "vendeur" in query_lower or "fournisseur" in query_lower:
            vendor_summary = self.server.get_vendor_summary()
            return f"RÉSUMÉ VENDEURS:\n{json.dumps(vendor_summary, indent=2, ensure_ascii=False)}"

        elif "catégorie" in query_lower or "type" in query_lower:
            category_summary = self.server.get_category_summary()
            return f"RÉSUMÉ CATÉGORIES:\n{json.dumps(category_summary, indent=2, ensure_ascii=False)}"

        return None

    def get_response(self, user_query, conversation_history=[]):
        """Get structured response from Groq LLM"""
        # Check for specific queries first
        specific_data = self._handle_specific_queries(user_query)

        # Create context
        context = self._create_context(user_query)

        # Prepare messages
        messages = [
            {"role": "system", "content": context}
        ]

        # Add conversation history (last 4 messages only)
        if conversation_history:
            messages.extend(conversation_history[-4:])

        # Add specific data if available
        if specific_data:
            messages.append({"role": "system", "content": specific_data})

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération de la réponse: {str(e)}"


class MCPHost:
    """MCP Host - Audit and permission layer"""

    def __init__(self, groq_api_key, log_file="mcp_log.txt"):
        self.client = MCPClient(groq_api_key)
        self.log_file = log_file
        self.authenticated = True  # Simple auth simulation (our streamlit dashboard does not handel authentification...)

    def _log_interaction(self, user_query, llm_response):
        """Log all interactions for audit"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] USER: {user_query}\n[{timestamp}] AI: {llm_response}\n---\n"

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Erreur lors de l'écriture du log: {e}")

    def process_query(self, user_query, conversation_history=[]):
        """Main entry point - processes query with permissions and logging"""
        if not self.authenticated:
            return "Accès non autorisé. Veuillez vous authentifier."

        # Get response from MCP Client
        response = self.client.get_response(user_query, conversation_history)

        # Log the interaction
        self._log_interaction(user_query, response)

        return response 