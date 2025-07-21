"""
Recommendation generation for supply chain issues.

This module generates actionable recommendations for detected issues.
"""
import os
import time
import json
import logging
import pandas as pd
import requests
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendationGenerator:
    """
    Generates actionable recommendations for supply chain anomalies.
    
    This class provides both rule-based and LLM-based approaches to generate
    recommendations for detected supply chain issues, with a focus on providing
    actionable insights for different stakeholder levels.
    
    Attributes:
        use_llm (bool): Whether to use LLM for enhanced recommendations
        api_key (str): API key for LLM service
        llm_model (str): Model name for LLM API
        high_priority_threshold (float): Threshold for high priority issues
    """
    
    def __init__(self, use_llm=None, api_key=None, llm_model="gpt-4o", high_priority_threshold=0.7):
        """
        Initialize the recommendation generator.
        
        Args:
            use_llm (bool, optional): Whether to use LLM for recommendations.
                If None, will check for API key in environment
            api_key (str, optional): API key for LLM service.
                If None, will check environment variable
            llm_model (str, optional): Model name to use for LLM API.
                Defaults to "gpt-4o"
            high_priority_threshold (float, optional): Threshold for high priority.
                Defaults to 0.7
        """
        # Check for API key in environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        # Determine if we should use LLM based on API key availability
        if use_llm is None:
            use_llm = api_key is not None
        
        self.use_llm = use_llm and api_key is not None
        self.api_key = api_key
        self.llm_model = llm_model
        self.high_priority_threshold = high_priority_threshold
        
        if self.use_llm:
            logger.info(f"Recommendation generator initialized with LLM support using {llm_model}")
        else:
            logger.info("Recommendation generator initialized with rule-based recommendations only")

    def generate_recommendations(self, anomalies):
        """
        Generate recommendations for identified anomalies.
        
        This method processes each anomaly and generates recommendations based on
        issue type, priority, and available data, using either rule-based or LLM
        approaches depending on configuration.
        
        Args:
            anomalies (pd.DataFrame): DataFrame containing anomalies with issue types
            
        Returns:
            pd.DataFrame: Anomalies with added recommendation columns
        """
        if anomalies.empty:
            logger.info("No anomalies to generate recommendations for.")
            return pd.DataFrame()

        # Deep copy the anomalies DataFrame to avoid modifying the original
        result_df = anomalies.copy()
        
        # Initialize new columns for recommendations
        result_df['Issue_Type'] = result_df['issue_type']  # Copy the issue_type to a cleaner column name
        result_df['Rule_Based_Recommendation'] = ""
        result_df['LLM_Recommendation'] = ""
        result_df['Final_Recommendation'] = ""  # This will contain either basic or LLM based on priority
        
        # Determine priority based on anomaly score if not already present
        if 'Priority' not in result_df.columns:
            result_df['Priority'] = result_df['anomaly_score'].apply(
                lambda x: 'High' if x > self.high_priority_threshold else 'Medium'
            )
        
        # Process each row individually
        for idx, row in result_df.iterrows():
            # Generate basic rule-based recommendation for all anomalies
            basic_recommendation = self.generate_rule_based_recommendation(row)
            result_df.at[idx, 'Rule_Based_Recommendation'] = basic_recommendation
            
            # Get the issue type and priority
            issue_type = row.get('issue_type', 'Unknown Issue')
            is_high_priority = row.get('Priority') == 'High'
            
            # Generate LLM recommendation if:
            # 1. It's a high-priority issue, OR
            # 2. It's an unclassified anomaly ('No_Issue')
            # AND we have LLM capability available
            if self.use_llm and (is_high_priority or issue_type == 'No_Issue'):
                try:
                    llm_recommendation = self.generate_llm_recommendation(row)
                    result_df.at[idx, 'LLM_Recommendation'] = llm_recommendation
                    result_df.at[idx, 'Final_Recommendation'] = llm_recommendation  # Use LLM
                    
                    logger.info(f"Generated LLM recommendation for {row.get('SKU', 'Unknown')} in {row.get('Country', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Error generating LLM recommendation: {str(e)}")
                    result_df.at[idx, 'Final_Recommendation'] = basic_recommendation  # Fallback to basic
            else:
                # For low-priority or when no LLM is available, use the basic recommendation
                result_df.at[idx, 'Final_Recommendation'] = basic_recommendation
        
        logger.info(f"Generated recommendations for {len(result_df)} anomalies")
        return result_df

    def generate_rule_based_recommendation(self, row_data):
        """
        Generate a rule-based recommendation based on issue type.
        
        This method creates specific recommendations for each type of supply chain issue
        using predefined business rules, without requiring external API calls.
        
        Args:
            row_data (dict or pd.Series): Data for a single anomaly
            
        Returns:
            str: A specific recommendation based on the issue type
        """
        # Ensure we're working with a dictionary
        if isinstance(row_data, pd.Series):
            row_data = row_data.to_dict()
        
        # Extract key information with safe defaults
        sku = row_data.get('product_number', 'Unknown SKU')
        country = row_data.get('reporter_country_code', 'Unknown Country')
        distributor = row_data.get('reporter_name', 'Unknown Distributor')
        issue = row_data.get('issue_type', 'Unknown Issue')
        
        # Generate recommendation based on issue type
        if issue == 'Inventory_Imbalance':
            # Safely get numerical values
            try:
                weeks_of_stock = float(row_data.get('WeeksOfStockT1', 0))
            except (ValueError, TypeError):
                weeks_of_stock = 0
                    
            if weeks_of_stock > 8:
                return f"Reduce inventory for distributor {distributor} at SKU {sku} in {country}. Current stock level of {weeks_of_stock:.1f} weeks exceeds target range (4-6 weeks). Consider promotions to increase sell-through and temporarily reduce ordering by 30%."
            else:
                return f"Increase inventory for distributor {distributor} at {sku} in {country}. Current stock level is only {weeks_of_stock:.1f} weeks, below minimum threshold of 4 weeks. Expedite shipments and increase order quantities by 25% for the next cycle."
        
        elif issue == 'Sales_Performance_Gap':
            try:
                target_achievement = float(row_data.get('TargetAchievement', 0))
            except (ValueError, TypeError):
                target_achievement = 0
                    
            return f"Review sales targets for distributor {distributor} at {sku} in {country}. Current achievement is only {target_achievement:.1f}% against target. Implement sales incentives and targeted marketing campaigns to improve performance by at least 15% in the next 30 days."
        
        elif issue == 'Shipments_Performance_Gap':
            try:
                target_achievement = float(row_data.get('TargetAchievement_ship', 0))
            except (ValueError, TypeError):
                target_achievement = 0
                    
            return f"Evaluate delivery performance for distributor {distributor} at {sku} in {country}. Current shipment volume is only {target_achievement:.1f}% against target.  of planned targets. Implement logistics optimizations and supplier coordination measures to increase shipment capacity by at least 15% in the next 30 days."
        
        elif issue == "Pricing_Issue":
            try:
                price_positioning = float(row_data.get('PricePositioning', 0))
            except (ValueError, TypeError):
                price_positioning = 0
                    
            return f"Adjust pricing strategy for distributor {distributor} at {sku} in {country}. Current price positioning of {price_positioning:.1f}% is making the product uncompetitive. Conduct competitive analysis and consider 5-10% tactical price adjustment in key segments."
        
        elif issue == "Supply_Chain_Disruption":
            try:
                backlog = int(row_data.get('Backlog', 0))
            except (ValueError, TypeError):
                backlog = 0
                    
            return f"Address backlog issues for distributor {distributor} at {sku} in {country}. Current backlog of {backlog} units is impacting fulfillment. Prioritize shipments, find alternative supply routes, and implement daily supply chain war room until resolved."
        
        elif issue == "Sell_Through_Bottleneck":
            try:
                sell_thru_ratio = float(row_data.get('SellThruToRatio', 0))
            except (ValueError, TypeError):
                sell_thru_ratio = 0
                    
            return f"Channel partner for distributor {distributor} at {sku} in {country} is not selling to end customers efficiently. Sell-through ratio is only {sell_thru_ratio:.2f} (target >0.85). Provide sales training, marketing support, and adjust partner incentives to reward sell-through activity."
        

        elif issue == "Aged Inventory":
            try:
                aged_inventory_pct = float(row_data.get('AgedInventoryPct', 0))
            except (ValueError, TypeError):
                aged_inventory_pct = 0
                    
            return f"Address aged inventory for distributor {distributor} at {sku} in {country}. Current aged inventory percentage is {aged_inventory_pct:.1f}%. Implement clearance promotions, bundle offers, and targeted marketing to reduce aged stock by at least 20% in the next quarter."
        
        elif issue == "Unknown" or issue == "Unknown Issue":
            # Enhanced recommendation for anomalies without a specific issue type
            # Extract key metrics to provide some context
            try:
                anomaly_score = float(row_data.get('anomaly_score', 0))
                sell_thru = float(row_data.get('SellThru', 0))
                sell_to = float(row_data.get('SellTo', 0))
                target_achievement = float(row_data.get('TargetAchievement', 0))
                inventory_turnover = float(row_data.get('InventoryTurnoverRate', 0))
                weeks_of_stock = float(row_data.get('WeeksOfStockT1', 0))
            except (ValueError, TypeError):
                # If metrics extraction fails, use a generic message
                return f"Investigate unclassified anomaly for distributor {distributor} at {sku} in {country}. The system has detected unusual patterns that don't match known issue types but may require attention."
            
            # Provide a more informative recommendation based on extracted metrics
            return (
                f"Investigate unclassified anomaly for distributor {distributor} at {sku} in {country} with anomaly score of {anomaly_score:.2f}. "
                f"Key metrics to examine: Sell-Through ({sell_thru:.0f} units), Sell-To ({sell_to:.0f} units), "
                f"Target Achievement ({target_achievement:.1f}%), Inventory Turnover ({inventory_turnover:.2f}), "
                f"Weeks of Stock ({weeks_of_stock:.1f}). Consider performing detailed analysis to identify potential emerging issues."
            )
            
        else: 
            return f"Investigate anomaly for SKU {sku} in {country}. The system has detected abnormal patterns that require further analysis."
            
    def generate_llm_recommendation(self, row_data):
        """
        Generate an enhanced recommendation using LLM.
        
        This method creates detailed, CEO-focused recommendations using an LLM API.
        For unclassified anomalies, it asks the LLM to perform exploratory analysis.
        
        Args:
            row_data (dict or pd.Series): Data for a single anomaly
            
        Returns:
            str: An enhanced recommendation from the LLM
        """
        if not self.use_llm or not self.api_key:
            raise ValueError("LLM recommendations not available. Check API key and settings.")
            
        # Ensure we're working with a dictionary
        if isinstance(row_data, pd.Series):
            row_data = row_data.to_dict()
        
        # Extract key information with safe defaults
        sku = row_data.get('SKU', 'Unknown SKU')
        country = row_data.get('Country', 'Unknown Country')
        distributor = row_data.get('reporter_name', 'Unknown Distributor')
        issue = row_data.get('issue_type', 'Unknown Issue')
        
        # Create a structured historical context with only available fields
        historical_context = {}
        
        # Helper function to safely convert values to JSON-serializable types
        def safe_value(val):
            if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
                return int(val)
            elif isinstance(val, (np.float64, np.float32, np.float16)):
                return float(val)
            elif isinstance(val, pd.Series):
                return val.to_dict()
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif pd.isna(val):
                return None
            return val
        
        # Build context sections
        sections = {
            'Sales Data': ['SellThru', 'SellTo', 'TargetQty'],
            'Inventory Status': ['T2Inventory','DistributorInventory','AgedInventory',
            'WeeksOfStockT1', 'WeeksOfStockT2'],
            'Supply Chain': ['Shipments', 'SupplyChainEfficiency'],
            'Market Position': ['NumCompetitors', 'PricePositioning'],
            'Derived Metrics': ['SellThruToRatio', 'InventoryTurnoverRate', 'TargetAchievement','AgedInventoryPct']
        }
        
        # Populate context with available data
        for section, fields in sections.items():
            section_data = {}
            for field in fields:
                if field in row_data and row_data[field] is not None:
                    try:
                        section_data[field] = safe_value(row_data[field])
                    except:
                        # Skip fields that can't be converted
                        pass
            if section_data:
                historical_context[section] = section_data
        
        # Convert context to JSON, handling any serialization issues
        try:
            context_json = json.dumps(historical_context, indent=2)
        except TypeError as e:
            # Fallback to a simpler string format if JSON conversion fails
            logger.error(f"JSON serialization error: {str(e)}")
            context_str = ""
            for section, data in historical_context.items():
                context_str += f"\n{section}:\n"
                for key, value in data.items():
                    context_str += f"  - {key}: {str(value)}\n"
            context_json = context_str
        
        # Create the prompt for the LLM - different prompt for unclassified anomalies
        if issue == 'No_Issue':
            prompt = f"""
            You are an expert supply chain analyst tasked with investigating an unclassified anomaly.
            
            This data point has been flagged as unusual by our anomaly detection system, but doesn't fit into
            our standard issue categories. Please analyze the data and identify potential issues or concerns.

            Product: {sku}
            Market: {country}
            Distributor: {distributor}
            
            Relevant metrics:
            {context_json}

            As a senior supply chain advisor, please provide:
            
            1. ANOMALY ANALYSIS: Carefully analyze the metrics to identify what makes this data point unusual.
            Look for values that are outliers or unexpected combinations of metrics.
            
            2. POTENTIAL ISSUE CATEGORIZATION: Based on your analysis, what type of issue might this be? 
            Consider both standard categories (Inventory Imbalance, Sales Performance Gap, Pricing Issue, 
            Supply Chain Disruption, Sell-Through Bottleneck) and potential new categories.
            
            3. ROOT CAUSE HYPOTHESIS: What are 2-3 possible root causes for this anomaly?
            
            4. RECOMMENDED INVESTIGATION: Specific steps the supply chain team should take to further
            investigate this anomaly.
            
            5. BUSINESS IMPLICATIONS: What could be the potential business impact if this anomaly 
            represents a real issue?

            Format your response as a concise executive advisory. 
            Include specific numbers from the data to support your analysis.
            """
        else:
            # Standard prompt for classified issues
            prompt = f"""
            You are an expert supply chain strategic advisor presenting critical insights to a CEO.
            
            You need to provide an executive-level recommendation for the following supply chain issue:

            Product: {sku}
            Market: {country}
            Distributor: {distributor}
            Issue Type: {issue}

            Relevant metrics:
            {context_json}

            As the CEO's trusted supply chain advisor, please provide:
            
            1. EXECUTIVE SUMMARY: A brief (2-sentence) summary of the critical issue
            
            2. ROOT CAUSE ANALYSIS: The most likely underlying causes of this issue based on the data
            
            3. STRATEGIC RECOMMENDATIONS: 2-3 specific executive-level actions that address the root causes
            
            4. BUSINESS IMPACT: The specific financial and operational impacts of implementing these recommendations
            
            5. KEY PERFORMANCE INDICATORS: The 1-2 most critical metrics the CEO should monitor to track improvement
            
            Format your response as a concise executive advisory with clear section headings. 
            Focus on strategic business implications rather than technical supply chain details.
            Include specific numbers and percentages from the data to support your analysis.
            """

        # Set up API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "You are a strategic supply chain advisor to the CEO."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }

        # Make API call with retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    return response_data['choices'][0]['message']['content']
                else:
                    raise ValueError("Unexpected response format from OpenAI API")
                
            except RequestException as e:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"API request failed, retrying in {sleep_time} seconds... ({str(e)})")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise
