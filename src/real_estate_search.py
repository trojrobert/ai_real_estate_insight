from firecrawl import FirecrawlApp
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
import time

def search_real_estate_with_firecrawl(location, min_price, max_price):
    """
    Use Firecrawl to search for real estate data based on location and price range.
    Returns a DataFrame of properties with focus on address and price.
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("FIRECRAWL_API_KEY")
        
        if not api_key:
            st.error("FIRECRAWL_API_KEY not found in environment variables. Please check your .env file.")
            return pd.DataFrame()
            
        # Initialize FireCrawl client with API key
        firecrawl = FirecrawlApp(api_key=api_key)
        
        # Define query - simplified to focus on address and price
        query = f"Find real estate properties for sale in {location} with prices between ${min_price} and ${max_price}. Include only property address and price"
        
        # Define research parameters
        params = {
            "maxDepth": 4,
            "timeLimit": 60,
            "maxUrls": 5,
            "focusTopics": ["real estate", "property listings", location]
        }
        
        # Progress tracking
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_value = 0
        activities = []
        
        # Activity callback
        def on_activity(activity):
            nonlocal progress_value, activities
            activities.append(activity)
            
            # Update the progress based on activity type
            if activity['type'] == 'searching':
                progress_value = min(60, progress_value + 5)
            elif activity['type'] == 'analyzing':
                progress_value = min(90, progress_value + 10)
            elif activity['type'] == 'complete':
                progress_value = 100
                
            progress_bar.progress(progress_value)
            progress_placeholder.text(f"Status: {activity['message']}")
            
        # Run deep research
        results = firecrawl.deep_research(
            query=query,
            params=params,
            on_activity=on_activity
        )
        
        # Process results into a DataFrame, focusing only on address and price
        properties = []
        
        # Extract property listings from research results - simplified
        for item in results.get("properties", []):
            property_data = {
                "address": item.get("address", ""),
                "neighborhood": item.get("neighborhood", ""),
                "price": item.get("price", 0),
            }
            properties.append(property_data)
        
        return pd.DataFrame(properties)
        
    except Exception as e:
        st.error(f"Error searching for real estate data: {str(e)}")
        return pd.DataFrame()

def get_average_rent_in_area(location, property_type="apartment"):
    """
    Use Firecrawl to get the average rent price in a given area.
    
    Args:
        location: String representing the location (city, zip, etc.)
        property_type: Type of property for rental data
        
    Returns:
        Float representing the average monthly rent in dollars
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("FIRECRAWL_API_KEY")
        
        if not api_key:
            st.error("FIRECRAWL_API_KEY not found in environment variables. Please check your .env file.")
            return 0
            
        # Initialize FireCrawl client with API key
        firecrawl = FirecrawlApp(api_key=api_key)
        
        # Placeholder message
        st.info(f"Researching rental prices in {location} for {property_type}s...")
        
        # Define query specific to rental listings
        query = f"Find average monthly rental prices for {property_type}s in {location}. Focus on current rental listings and typical rent ranges."
        
        # Define research parameters
        params = {
            "maxDepth": 3,
            "timeLimit": 30,
            "maxUrls": 4,
            "focusTopics": ["rental properties", "apartment rentals", "rental market", location]
        }
        
        # Progress tracking
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_value = 0
        
        # Activity callback
        def on_activity(activity):
            nonlocal progress_value
            
            # Update the progress based on activity type
            if activity['type'] == 'searching':
                progress_value = min(60, progress_value + 10)
            elif activity['type'] == 'analyzing':
                progress_value = min(90, progress_value + 15)
            elif activity['type'] == 'complete':
                progress_value = 100
                
            progress_bar.progress(progress_value)
            progress_placeholder.text(f"Rental research: {activity['message']}")
        
        # Run deep research
        results = firecrawl.deep_research(
            query=query,
            params=params,
            on_activity=on_activity
        )
        
        # Process results to extract rental prices
        rental_prices = []
        
        # Check if rental information exists in the results
        rental_data = results.get("rentalData", [])
        if rental_data:
            for item in rental_data:
                if property_type.lower() in item.get("propertyType", "").lower():
                    price = item.get("price", 0)
                    if price > 0:
                        rental_prices.append(price)
        
        # If no direct rental data, try to extract from the summary
        if not rental_prices:
            summary = results.get("summary", "")
            # Try to extract rental prices from text using regex
            import re
            price_matches = re.findall(r'\$(\d{1,3}(?:,\d{3})*)', summary)
            for match in price_matches:
                price = float(match.replace(',', ''))
                # Only consider reasonable rental prices (between $500 and $10,000)
                if 500 <= price <= 10000:
                    rental_prices.append(price)
        
        # Clear the progress indicators
        progress_placeholder.empty()
        progress_bar.empty()
        
        # Calculate average rent
        if rental_prices:
            avg_rent = sum(rental_prices) / len(rental_prices)
            st.success(f"Found rental data for {location}")
            return avg_rent
        else:
            # Fall back to mock data if no rental prices found
            st.warning(f"Could not find specific rental data for {location}. Using estimates.")
            mock_rental_data = {
                "Austin, TX": {"apartment": 1550, "house": 2200},
                "New York, NY": {"apartment": 3500, "house": 5000},
                "San Francisco, CA": {"apartment": 3200, "house": 4800},
                "Chicago, IL": {"apartment": 1800, "house": 2600},
                "Miami, FL": {"apartment": 2100, "house": 3000},
            }
            default_rent = {"apartment": 1500, "house": 2400}
            location_data = mock_rental_data.get(location, default_rent)
            return location_data.get(property_type, default_rent[property_type])
            
    except Exception as e:
        st.warning(f"Error retrieving rental data: {str(e)}")
        return 0 