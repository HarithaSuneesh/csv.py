import streamlit as st
import pymysql
import pandas as pd

# 1. Use st.cache_resource so you don't reconnect on every button click
@st.cache_resource
def get_connection():
    return pymysql.connect(
        host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
        port=4000,
        user="2LQdqxSJhDf5xi1.root",
        password="Av6yG6tlEREJUg0M",
        database="eatdata",
        ssl={"ca":None},
        autocommit=True
    )

def execute_query(query, params=None):
    # Get the cached connection
    conn = get_connection()
    try:
        # Using read_sql with the connection object
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()

#streamlit UI
st.title("Uber Eats Bangalore Restaurant Intelligence & Decision Support System")

#sidebar filters
st.sidebar.header("Filters")


query = "SELECT name, rate, location,approx_cost_for_two_people FROM ubereats WHERE 1=1"
   
import streamlit as st
import pandas as pd

# 1. Define UI Widgets in the Sidebar
st.sidebar.header("Intelligence Filters")

# Location Multiselect
available_locations = ["All", "Indiranagar", "Koramangala", "Jayanagar", "HSR", "BTM"]
locations = st.sidebar.multiselect("Select Locations", available_locations, default="All")

# Rating Slider
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 2.5, 0.1)

# Price Segment Selectbox
price_segments = st.sidebar.selectbox("Price Tier", ["All", "Budget", "Mid-Range", "Premium"])

# Feature Checkboxes
# --- Corrected Sidebar Feature Filters ---
col1, col2 = st.sidebar.columns(2)
# Adding unique keys prevents the DuplicateElementId error
online_order = col1.checkbox("Online Order", key="online_filter_key")
book_table = col2.checkbox("Book Table", key="book_table_filter_key")

# 2. DYNAMIC SQL CONSTRUCTION
# Start with a base query that is always true (1=1)
query = "SELECT name, rate, location,approx_cost_for_two_people, cuisines FROM ubereats WHERE 1=1"

# Apply Filter 1: Location
if locations and "All" not in locations:
    location_list = "', '".join(locations)
    query += f" AND location IN ('{location_list}')"

# Apply Filter 2: Rating
if min_rating > 0:
    query += f" AND rate >= {min_rating}"

# Apply Filter 3: Price
if price_segments == "Budget":
    query += " AND approx_cost_for_two_people < 300"
elif price_segments == "Mid-Range":
    query += " AND approx_cost_for_two_people BETWEEN 300 AND 800"
elif price_segments == "Premium":
    query += " approx_cost_for_two_people > 800"

# Apply Filter 4: Features
if online_order:
    query += " AND online_order = 'Yes'"
if book_table:
    query += " AND book_table = 'Yes'"


# 3. EXECUTION AND DISPLAY
if st.sidebar.button("Apply Filters"):
    # Using your existing execute_query function
    df = execute_query(query)
    
    if not df.empty:
      st.subheader(f"Found {len(df)} ubereats")
      st.dataframe(df, use_container_width=True)
        
        # Show the generated SQL for your project documentation
      with st.expander("Show SQL Query"):
         st.code(query)
    else:
      st.warning("No results found. Try loosening your filters!")


#execute and display sql queries
st.subheader("Resturant Analysis")
queries = {
    "1.Bangalore locations have the highest average restaurant ratings": "SELECT location,AVG(rate) AS average_rating FROM ubereats GROUP BY location ORDER BY average_rating DESC LIMIT 10;",
    "2.Bangalore Locations Over-saturated with Restaurant" :"SELECT location,COUNT(name) AS restaurant_count FROM ubereats GROUP BY location ORDER BY restaurant_count DESC LIMIT 20;",
    "3.Average Restaurant Ratings by Online Ordering Availability" : "SELECT online_order, AVG(rate) AS average_rating FROM ubereats GROUP BY online_order",
    "4.Average ratings for table booking vs. no table booking" : "SELECT book_table,AVG(rate) AS average_rating FROM ubereats GROUP BY book_table",
    "5.price range delivers the best customer satisfaction"  : "SELECT approx_cost_for_two_people,rate FROM ubereats WHERE rate IS NOT NULL;",
    "7.cuisines are most common in Bangalore" : "SELECT cuisines, COUNT(name) AS restaurant_count FROM ubereats GROUP BY cuisines ORDER BY restaurant_count DESC LIMIT 10;",
    "8.cuisines receive the highest average ratings" : "SELECT cuisines, AVG(rate) AS average_rating FROM ubereats GROUP BY  cuisines ORDER BY   average_rating DESC LIMIT 10;",
    "11.locations are ideal for premium restaurant onboarding" : "SELECT location, AVG(rate) AS average_rating, COUNT(name) AS restaurant_count FROM ubereats GROUP BY location;",
    "13.restaurants offering both online ordering and table booking perform better" : "SELECT online_order,  book_table, AVG(rate) AS average_rating FROM ubereats GROUP BY online_order, book_table ORDER BY average_rating DESC;",
    "14. restaurants are top performers within each pricing segment" : "SELECT name,rate, approx_cost_for_two_people FROM ubereats WHERE rate IS NOT NULL"

}

#dropdown to select a query
selected_query=st.selectbox("Select a Query: ", list(queries.keys()))

#button to execute the selected query
if st.button("Run Query"):
    with st.spinner("FetchingData .."):
        df= execute_query(queries[selected_query])
        st.success("Query Executed Successfully")
        st.dataframe(df)


#streamlit UI
st.title("Order Eats Data")

st.write("What is the total order value for each restaurant?")
st.dataframe(execute_query("SELECT restaurant_name, SUM(order_value) AS total_revenue FROM orders_data GROUP BY restaurant_name  ORDER BY total_revenue DESC;"))

st.write("Which payment method is most frequently used?")
st.dataframe(execute_query("SELECT payment_method, COUNT(order_id) AS number_of_orders FROM orders_data GROUP BY payment_method ORDER BY number_of_orders DESC;"))

st.write("What is the average order value across all orders?")
st.dataframe(execute_query("SELECT AVG(order_value) AS average_order_value FROM orders_data"))

st.write("How many orders used a discount vs. no discount?")
st.dataframe(execute_query("SELECT CASE WHEN discount_used = 1 THEN 'Yes' ELSE 'No' END AS discount_status, COUNT(order_id) AS number_of_orders FROM orders_data GROUP BY discount_status"))

st.write("Which are the top 5 restaurants by total number of orders?")
st.dataframe(execute_query("SELECT restaurant_name, COUNT(order_id) AS total_orders FROM orders_data GROUP BY restaurant_name ORDER BY total_orders DESC LIMIT 5;"))