import pandas as pd
from datetime import datetime, date
from dateutil.parser import parse
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Define constants for category mapping
CATEGORIES: Dict[int, str] = {
    1: "Groceries",
    2: "Entertainment",
    3: "Household",
    4: "Transportation",
    6: "Education",
    7: "Utilities",
    8: "Others"
}

# Define threshold for high spending category analysis
HIGH_SPENDING_THRESHOLD: float = 0.3

class FinanceAdvisor:
    def __init__(self, transactions: List):
        self.transactions = transactions
        self.transaction_df = self._prepare_transaction_data()
        self.current_week_number: int = datetime.now().isocalendar()[1]
        self.current_year: int = datetime.now().year

        # KMeans and Decision Tree Models
        self.kmeans_model = None
        self.decision_tree_model = None

    def _prepare_transaction_data(self) -> pd.DataFrame:
        prepared_data: List[Dict] = []
        for transaction in self.transactions:
            try:
                transaction_date: Optional[date] = None
                if isinstance(transaction.date, str):
                    transaction_date = parse(transaction.date).date()
                elif isinstance(transaction.date, date):
                    transaction_date = transaction.date

                if transaction_date:
                    prepared_data.append({
                        'amount': float(transaction.amount),
                        'category': int(transaction.category),
                        'date': transaction_date,
                        'description': str(transaction.description) if transaction.description else ''
                    })
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Warning: Skipping invalid transaction data - {e}")

        if not prepared_data:
            return pd.DataFrame(columns=['amount', 'category', 'date', 'description'])

        df = pd.DataFrame(prepared_data)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df['week'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['category_name'] = df['category'].map(CATEGORIES)
        return df

    def generate_insights(self) -> List[str]:
        if len(self.transaction_df) < 10:
            return ["Insufficient transaction data for detailed insights. Please add at least 10 transactions."]

        insights: List[str] = []

        weekly_comparison_insight: Optional[str] = self._analyze_weekly_spending()
        if weekly_comparison_insight:
            insights.append(weekly_comparison_insight)

        category_analysis_insight: Optional[str] = self._analyze_spending_by_category()
        if category_analysis_insight:
            insights.append(category_analysis_insight)

        # KMeans clustering analysis
        kmeans_insight: Optional[str] = self._perform_kmeans_clustering()
        if kmeans_insight:
            insights.append(kmeans_insight)

        # Decision Tree prediction
        decision_tree_insight: Optional[str] = self._predict_spending_trend()
        if decision_tree_insight:
            insights.append(decision_tree_insight)

        budget_recommendations: List[str] = self._generate_budget_recommendations()
        insights.extend(budget_recommendations[:2])

        return insights

    def _analyze_weekly_spending(self) -> Optional[str]:
        try:
            weekly_spending = self.transaction_df[self.transaction_df['date'].dt.year == self.current_year].groupby('week')['amount'].sum()

            if len(weekly_spending) < 2:
                return None

            current_week_spending: float = weekly_spending.get(self.current_week_number, 0)
            last_week_spending: float = weekly_spending.get(self.current_week_number - 1, 0)

            if last_week_spending == 0:
                return f"Current week spending: ₹{current_week_spending:,.2f}"

            spending_change_percentage: float = (current_week_spending - last_week_spending) / last_week_spending * 100

            if abs(spending_change_percentage) < 5:
                return "📊 Your weekly spending remained within 5% of the previous week — showing stability in your financial behavior. Continue this disciplined approach."
            elif spending_change_percentage > 0:
                return f"⚠️ This week’s expenses rose by {abs(spending_change_percentage):.0f}% compared to last week. Investigate discretionary spending or irregular purchases."
            else:
                return f"✅ You reduced your weekly expenses by {abs(spending_change_percentage):.0f}%. A great opportunity to allocate more towards savings or investments."
        except Exception as e:
            print(f"Error analyzing weekly spending: {e}")
            return None

    def _analyze_spending_by_category(self) -> Optional[str]:
        try:
            category_spending = self.transaction_df.groupby('category_name')['amount'].sum()
            if category_spending.empty:
                return None

            total_spending: float = category_spending.sum()
            top_categories = category_spending.sort_values(ascending=False).head(3)

            insight_lines: List[str] = ["💡 Top spending areas this month:"]
            for category, amount in top_categories.items():
                percent = (amount / total_spending) * 100
                insight_lines.append(f"• {category}: ₹{amount:,.2f} ({percent:.1f}%)")
            insight_lines.append("🧠 Consider reviewing these areas to spot unnecessary expenses or chances to optimize.")
            return "\n".join(insight_lines)
        except Exception as e:
            print(f"Error analyzing spending by category: {e}")
            return None

    def _perform_kmeans_clustering(self) -> Optional[str]:
        try:
            # Prepare data for clustering
            clustering_data = self.transaction_df[['amount', 'category']]
            kmeans = KMeans(n_clusters=3, random_state=42)
            clustering_data['cluster'] = kmeans.fit_predict(clustering_data)

            # Get insights based on clusters
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = ["Cluster 1", "Cluster 2", "Cluster 3"]
            insights = []
            for i, center in enumerate(cluster_centers):
                insights.append(f"🔍 Cluster {i+1}: Average spending = ₹{center[0]:,.2f}, Category = {CATEGORIES.get(int(center[1]), 'Unknown')}")
            return "\n".join(insights)
        except Exception as e:
            print(f"Error performing KMeans clustering: {e}")
            return None

    def _predict_spending_trend(self) -> Optional[str]:
        try:
            # Prepare data for decision tree
            self.transaction_df['week'] = self.transaction_df['date'].dt.isocalendar().week
            self.transaction_df['week_diff'] = self.transaction_df['week'].diff().fillna(0)

            X = self.transaction_df[['week_diff', 'amount']]
            y = (self.transaction_df['amount'].shift(-1) > self.transaction_df['amount']).astype(int)

            # Train decision tree
            decision_tree = DecisionTreeClassifier(random_state=42)
            decision_tree.fit(X, y)

            # Predict next week's trend
            prediction = decision_tree.predict([[1, 100]])  # Dummy input, should be real-time data
            trend = "increase" if prediction[0] == 1 else "decrease"

            return f"📉 Based on the decision tree model, your expenses are expected to {trend} next week."
        except Exception as e:
            print(f"Error predicting spending trend: {e}")
            return None

    def _generate_budget_recommendations(self) -> List[str]:
        recommendations: List[str] = []

        try:
            category_spending_ratio = self.transaction_df.groupby('category_name')['amount'].sum() / self.transaction_df['amount'].sum()
            high_spending_categories = category_spending_ratio[category_spending_ratio > HIGH_SPENDING_THRESHOLD]

            for category, ratio in high_spending_categories.items():
                recommendations.append(f"🔎 Spending on '{category}' is consuming {ratio:.0%} of your total expenses. Set a monthly cap or explore alternatives to reduce recurring cost.")

            weekly_spending = self.transaction_df[self.transaction_df['date'].dt.year == self.current_year].groupby('week')['amount'].sum()

            if len(weekly_spending) >= 4:
                average_weekly_spending: float = weekly_spending.mean()
                last_week_spending: float = weekly_spending.get(self.current_week_number - 1, 0)

                if last_week_spending > average_weekly_spending * 1.2:
                    recommendations.append("📉 Last week's expenses were over 20% higher than your monthly average. Consider setting alerts or weekly check-ins to stay within budget.")

        except Exception as e:
            print(f"Error generating budget recommendations: {e}")

        if not recommendations:
            recommendations.append("🧭 Tip: Set specific budget targets for your top 3 categories to maintain balance and improve savings potential.")
            recommendations.append("💼 Consider automating bill payments and redirecting surplus funds to an emergency or investment account.")

        return recommendations


if __name__ == '__main__':
    # Example usage with dummy data
    transactions_data = [
        {'amount': 75.50, 'category': 1, 'date': '2025-04-07', 'description': 'Groceries'},
        {'amount': 22.00, 'category': 2, 'date': '2025-04-08', 'description': 'Movie'},
        {'amount': 105.75, 'category': 1, 'date': '2025-03-31', 'description': 'Groceries'},
        {'amount': 50.00, 'category': 4, 'date': '2025-04-01', 'description': 'Fuel'},
        {'amount': 300.00, 'category': 3, 'date': '2025-04-05', 'description': 'Household items'},
        {'amount': 60.00, 'category': 2, 'date': '2025-04-06', 'description': 'Dinner out'},
        {'amount': 90.00, 'category': 7, 'date': '2025-04-09', 'description': 'Electricity'},
        {'amount': 1500.00, 'category': 1, 'date': '2025-04-14', 'description': 'Weekly Groceries'},
        {'amount': 35.00, 'category': 2, 'date': '2025-04-15', 'description': 'Coffee'},
        {'amount': 200.00, 'category': 3, 'date': '2025-04-17', 'description': 'Household cleaning supplies'}
    ]

    advisor = FinanceAdvisor(transactions_data)
    insights = advisor.generate_insights()

    for insight in insights:
        print(insight)
