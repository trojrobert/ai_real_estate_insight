import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
from datetime import datetime
import time
import folium
import numpy as np
import random
from geopy.geocoders import Nominatim
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Real Estate Investment Research", layout="wide")

# Custom styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #2563EB;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #F3F4F6;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def search_real_estate(location, min_price, max_price):
    """
    Use Firecrawl to search for real estate data based on location and price range.
    Replace this with actual Firecrawl implementation.
    """
    try:
        # Mock API endpoint for Firecrawl (replace with actual API)
        api_endpoint = "https://api.firecrawl.com/search"
        
        # Prepare search parameters
        params = {
            "query": f"real estate for sale in {location} price range {min_price} to {max_price}",
            "type": "real_estate",
            "limit": 20
        }
        
        # In a real implementation, you would make an API call like:
        # response = requests.post(api_endpoint, json=params, headers={"Authorization": "Bearer YOUR_API_KEY"})
        # results = response.json()
        
        # For demo purposes, we'll generate mock data
        st.info("Searching for properties... (simulated for demonstration)")
        
        # Simulate API delay
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
        
        # Generate mock data based on inputs
        results = []
        property_types = ["Single Family", "Condo", "Townhouse", "Multi-Family"]
        neighborhoods = [f"{location} North", f"{location} South", f"{location} Downtown", 
                         f"{location} East", f"{location} West"]
        
        for i in range(15):
            price = random.randint(int(min_price), int(max_price))
            sqft = random.randint(800, 3500)
            results.append({
                "id": i,
                "address": f"{random.randint(100, 9999)} {random.choice(['Oak', 'Maple', 'Pine', 'Cedar'])} St",
                "neighborhood": random.choice(neighborhoods),
                "price": price,
                "square_feet": sqft,
                "bedrooms": random.randint(1, 5),
                "bathrooms": random.randint(1, 4),
                "property_type": random.choice(property_types),
                "year_built": random.randint(1950, 2023),
                "price_per_sqft": round(price / sqft, 2)
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Error searching for real estate data: {str(e)}")
        return pd.DataFrame()

def get_city_coordinates(city_name):
    """
    Get coordinates for a given city using geocoding.
    
    Args:
        city_name (str): Name of the city
        
    Returns:
        tuple: (latitude, longitude) of the city center
    """
    try:
        geolocator = Nominatim(user_agent="real_estate_app")
        location = geolocator.geocode(city_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not find coordinates for {city_name}, using default location")
            return (40.7128, -74.0060)  # Default to NYC
    except Exception as e:
        print(f"Error finding city coordinates: {e}. Using default location.")
        return (40.7128, -74.0060)  # Default to NYC

def generate_demo_data(city_name):
    """
    Generate demo real estate data for the specified city.
    
    Args:
        city_name (str): Name of the city
       
    Returns:
        tuple: (price_data_df, properties_list)
    """
    # Generate time series price data
    years = list(range(2010, 2023))
    
    # Randomize base prices based on city name (just for demo variation)
    seed = sum(ord(c) for c in city_name)
    random.seed(seed)
    
    base_downtown = random.randint(200000, 400000)
    base_suburbs = random.randint(150000, 300000)
    base_waterfront = random.randint(250000, 500000)
    
    # Generate price trends with some randomness
    downtown_prices = [int(base_downtown * (1 + 0.03*i + random.uniform(-0.02, 0.02))) for i in range(len(years))]
    suburbs_prices = [int(base_suburbs * (1 + 0.025*i + random.uniform(-0.015, 0.015))) for i in range(len(years))]
    waterfront_prices = [int(base_waterfront * (1 + 0.035*i + random.uniform(-0.02, 0.025))) for i in range(len(years))]
    
    # Create price DataFrame
    df = pd.DataFrame({
        'Year': years,
        'Downtown': downtown_prices,
        'Suburbs': suburbs_prices,
        'Waterfront': waterfront_prices
    })
    
    # Get city coordinates
    city_center = get_city_coordinates(city_name)
    
    # Generate property listings around the city center
    properties = []
    areas = ['Downtown', 'Suburbs', 'Waterfront']
    for _ in range(15):  # Generate 15 sample properties
        # Random location near city center
        lat_offset = random.uniform(-0.03, 0.03)
        lon_offset = random.uniform(-0.03, 0.03)
        
        area = random.choice(areas)
        if area == 'Downtown':
            price = random.randint(downtown_prices[-1] - 50000, downtown_prices[-1] + 50000)
            sqft = random.randint(800, 1400)
        elif area == 'Suburbs':
            price = random.randint(suburbs_prices[-1] - 40000, suburbs_prices[-1] + 40000)
            sqft = random.randint(1500, 2500)
        else:  # Waterfront
            price = random.randint(waterfront_prices[-1] - 60000, waterfront_prices[-1] + 100000)
            sqft = random.randint(1200, 1800)
        
        properties.append({
            "location": [city_center[0] + lat_offset, city_center[1] + lon_offset],
            "price": f"${price:,}",
            "area": area,
            "sqft": sqft,
            "bedrooms": random.randint(1, 4)
        })
    
    return df, properties

def create_price_trend_graph(price_data, city_name):
    """
    Create a line graph of real estate prices.
    
    Args:
        price_data (DataFrame): DataFrame with price data
        city_name (str): Name of the city for the title
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_data['Year'], price_data['Downtown'], marker='o', linewidth=2, label='Downtown')
    ax.plot(price_data['Year'], price_data['Suburbs'], marker='s', linewidth=2, label='Suburbs')
    ax.plot(price_data['Year'], price_data['Waterfront'], marker='^', linewidth=2, label='Waterfront')

    ax.set_title(f'Average Home Prices in {city_name} (2010-2022)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Price ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xticks(price_data['Year'][::2])  # Show every other year to avoid crowding

    # Add price annotations for the last year
    for area in ['Downtown', 'Suburbs', 'Waterfront']:
        ax.annotate(f"${price_data[area].iloc[-1]:,}", 
                  xy=(price_data['Year'].iloc[-1], price_data[area].iloc[-1]),
                  xytext=(10, 0), 
                  textcoords='offset points',
                  fontsize=9)

    return fig, ax

def create_property_map(properties, city_name):
    """
    Create a folium map with property markers.
    
    Args:
        properties (list): List of property dictionaries
        city_name (str): Name of the city for the title
        
    Returns:
        folium.Map: The folium map object
    """
    # Calculate the average location for the map center
    if properties:
        avg_lat = sum(prop["location"][0] for prop in properties) / len(properties)
        avg_lon = sum(prop["location"][1] for prop in properties) / len(properties)
    else:
        # If no properties, use the city coordinates directly
        city_coords = get_city_coordinates(city_name)
        avg_lat, avg_lon = city_coords
    
    # Create the map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>Real Estate Properties in {city_name}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add markers for each property
    for prop in properties:
        # Create popup with property info
        popup_text = f"""
        <strong>Price:</strong> {prop['price']}<br>
        <strong>Area:</strong> {prop['area']}<br>
        <strong>Size:</strong> {prop['sqft']} sq.ft.<br>
        <strong>Bedrooms:</strong> {prop['bedrooms']}
        """
        
        # Choose color based on area
        colors = {
            'Downtown': 'blue',
            'Suburbs': 'green',
            'Waterfront': 'red'
        }
        color = colors.get(prop['area'], 'gray')
        
        # Add marker
        folium.Marker(
            location=prop["location"],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 100px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px">
        <p><i class="fa fa-circle" style="color:blue"></i> Downtown</p>
        <p><i class="fa fa-circle" style="color:green"></i> Suburbs</p>
        <p><i class="fa fa-circle" style="color:red"></i> Waterfront</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Return the map object instead of saving it
    return m

def generate_insights(price_data, properties, city_name):
    """
    Generate insights from the real estate data.
    
    Args:
        price_data (DataFrame): DataFrame with price data
        properties (list): List of property dictionaries
        city_name (str): Name of the city
        
    Returns:
        str: Insights text
    """
    # Calculate price increases over time
    first_year = price_data['Year'].iloc[0]
    last_year = price_data['Year'].iloc[-1]
    years_span = last_year - first_year
    
    insights = []
    insights.append(f"# Real Estate Market Insights for {city_name}\n")
    
    # Price trend insights
    for area in ['Downtown', 'Suburbs', 'Waterfront']:
        first_price = price_data[area].iloc[0]
        last_price = price_data[area].iloc[-1]
        percent_increase = ((last_price - first_price) / first_price) * 100
        annual_increase = percent_increase / years_span
        
        insights.append(f"## {area} Area")
        insights.append(f"- Price in {first_year}: ${first_price:,}")
        insights.append(f"- Price in {last_year}: ${last_price:,}")
        insights.append(f"- Total increase: {percent_increase:.1f}% over {years_span} years")
        insights.append(f"- Average annual appreciation: {annual_increase:.1f}%\n")
    
    # Current market insights
    insights.append("## Current Market Summary")
    
    # Calculate average prices by area from current listings
    area_prices = {'Downtown': [], 'Suburbs': [], 'Waterfront': []}
    for prop in properties:
        area = prop['area']
        # Extract numeric price from the formatted string
        price_value = int(prop['price'].replace('$', '').replace(',', ''))
        area_prices[area].append(price_value)
    
    for area, prices in area_prices.items():
        if prices:
            avg_price = sum(prices) / len(prices)
            insights.append(f"- Average listing price in {area}: ${avg_price:,.0f}")
            insights.append(f"- Price per sq.ft in {area}: ${avg_price / 1500:,.0f}")
    
    return "\n".join(insights)

def main():
    # Header
    st.markdown('<div class="title">Real Estate Insight</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Uses AI to reasearch real estate and give you insights</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Single location input for both property search and analysis
        location = st.text_input("City", "Berlin")
    
    with col2:
        price_range = st.slider("Price Range ($)", 
                               min_value=50000, 
                               max_value=2000000, 
                               value=(200000, 800000),
                               step=10000)
    
    min_price, max_price = price_range
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Single research button
    search_button = st.button("Search Properties")
    
    if search_button:

        df = search_real_estate(location, min_price, max_price)
        
        st.success(f"Found {len(df)} properties in {location} within your price range!")
             
        
        with st.spinner(f"Generating market analysis for {location}..."):
            # Generate data
            price_data, properties = generate_demo_data(location)
            
            # Create visualizations
            fig, ax = create_price_trend_graph(price_data, location)
            folium_map = create_property_map(properties, location)
            
            # Display price trend graph directly using st.pyplot
            st.subheader(f"Real Estate Price Trends in {location}")
            st.pyplot(fig)
            
            # Display property map directly using components.html
            st.subheader(f"Property Map for {location}")
            
            # Convert the folium map to HTML and display it
            map_html = folium_map._repr_html_()
            st.components.v1.html(map_html, height=500)                      


            if not df.empty:
   
                
                # Visualization section
                st.subheader("Property Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price distribution by property type
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df.boxplot(column='price', by='property_type', ax=ax)
                    ax.set_title('Price Distribution by Property Type')
                    ax.set_ylabel('Price ($)')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    # Price vs. Square Footage
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df['square_feet'], df['price'], alpha=0.7)
                    ax.set_title('Price vs. Square Footage')
                    ax.set_xlabel('Square Feet')
                    ax.set_ylabel('Price ($)')
                    # Add a trend line
                    z = np.polyfit(df['square_feet'], df['price'], 1)
                    p = np.poly1d(z)
                    ax.plot(df['square_feet'], p(df['square_feet']), "r--", alpha=0.8)
                    st.pyplot(fig)
                
                # Additional charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price per square foot by neighborhood
                    fig, ax = plt.subplots(figsize=(10, 6))
                    neighborhood_avg = df.groupby('neighborhood')['price_per_sqft'].mean().sort_values(ascending=False)
                    neighborhood_avg.plot(kind='bar', ax=ax)
                    ax.set_title('Average Price per Sq. Ft. by Neighborhood')
                    ax.set_ylabel('Price per Sq. Ft. ($)')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    # Distribution of properties by bedrooms
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df['bedrooms'].value_counts().sort_index().plot(kind='bar', ax=ax)
                    ax.set_title('Number of Properties by Bedroom Count')
                    ax.set_xlabel('Bedrooms')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                
                # Investment metrics
                st.subheader("Investment Metrics")
                
                # Calculate and display some key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Price", f"${df['price'].mean():,.2f}")
                
                with col2:
                    st.metric("Average Price/Sq.Ft", f"${df['price_per_sqft'].mean():,.2f}")
                
                with col3:
                    st.metric("Median Home Size", f"{df['square_feet'].median():,.0f} sq.ft")
                
                # Display results
                st.subheader("Property Listings")
                st.dataframe(df)
                
                # Download options
                st.download_button(
                    label="Download Data as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"real_estate_{location.replace(', ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
            
            else:
                st.warning("No properties found. Try adjusting your search criteria.")
        
             # Generate and display insights
            st.subheader("Market Insights")
            insights = generate_insights(price_data, properties, location)
            st.markdown(insights)

            st.success(f"Analysis for {location} completed successfully!")       

               
if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        st.error("Missing required packages. Please install with: pip install numpy")
    
    main()

