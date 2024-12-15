"""
Final Project: Interactive Data Explorer with Fortune 500 Corporate Headquarters Data
Author: [Your Name]

This Streamlit application explores a dataset of Fortune 500 company headquarters.
It demonstrates various Python and data analysis features as required:

Python Features:
[PY1] A function with two or more parameters (one with a default), called multiple times.
[PY2] A function that returns more than one value.
[PY3] Error checking with try/except.
[PY4] A list comprehension.
[PY5] A dictionary usage (keys, values).

Streamlit Features:
[ST1], [ST2], [ST3] At least three different widgets.
[ST4] Customized page design features (sidebar, title, images).

Visualizations:
[VIZ1], [VIZ2], [VIZ3], [VIZ4], [VIZ5], [VIZ6], [VIZ7] Seven different charts/graphs/tables.
[MAP] A detailed PyDeck map with customization.

Data Analytics:
[DA1] Data cleaning/manipulation.
[DA2] Sorting data.
[DA3] Finding top largest/smallest values.
[DA4] Filter data by one condition.
[DA5] Filter data by multiple conditions.
[DA6] Pivot table analysis.
[DA7] Adding/dropping/selecting/creating new columns.
[DA8] Iterate through rows with iterrows().
[DA9] Perform calculations on DataFrame columns.
[DA10] Additional calculations for deeper insights.

The app includes multiple pages, accessible through a sidebar.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="Fortune 500 HQ Explorer",
    page_icon=":office:",
    layout="wide"
)


# [PY3]: Data Loading function and Error checking with try/except
def load_data(filename: str) -> pd.DataFrame:
    """
    [PY3] Load CSV file with error handling.
    If loading fails, return an empty DataFrame.
    """
    try:
        data = pd.read_csv(filename, encoding='utf-8')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# Load the primary dataset
df = load_data("fortune_500_hq.csv")

# Data Cleaning & Manipulation [DA1], [DA7], [DA9]
# Rename columns
df.rename(
    columns={
        'NAME': 'CompanyName',
        'CITY': 'City',
        'STATE': 'State',
        'EMPLOYEES': 'Employees',
        'REVENUES': 'Revenue',
        'LATITUDE': 'Latitude',
        'LONGITUDE': 'Longitude',
        'PROFIT': 'Profit'
    },
    inplace=True
)

# [DA7], [DA9], [DA10]: Creating new columns based on existing data
# Convert revenue to billions
df['RevenueBillions'] = df['Revenue'] / 1000.0
# Calculate profit margin as a percentage
df['ProfitMargin'] = df['Profit'] / df['Revenue'] * 100
# Calculate profit per employee
df['ProfitPerEmployee'] = df['Profit'] / df['Employees']
# Calculate revenue per employee
df['RevenuePerEmployee'] = df['Revenue'] / df['Employees']
# Calculate efficiency ratio (profit-to-revenue ratio)
df['EfficiencyRatio'] = df['Profit'] / df['Revenue']


# [PY5]: A dictionary usage (keys, values)
# Count occurrences of each state and store in a dictionary
state_counts_series = df['State'].value_counts()
state_counts = state_counts_series.to_dict()

# [PY4]: A list comprehension
# Create a list of tuples (CompanyName, City, State) for reference
company_list = [(name, city, state) for name, city, state in zip(df['CompanyName'], df['City'], df['State'])]


# [PY1]: A function with two or more parameters (one with a default), called multiple times
def filter_by_employees(dataframe: pd.DataFrame, min_emp: int = 0, max_emp: int = 500000) -> pd.DataFrame:
    """
    [PY1]: This function filters the dataframe by a range of Employees
    It is called multiple times with different parameters
    """
    filtered = dataframe[(dataframe['Employees'] >= min_emp) & (dataframe['Employees'] <= max_emp)]
    return filtered


# Calling the function with default values
default_emp_filter = filter_by_employees(df)

# Calling the function with custom values
custom_emp_filter = filter_by_employees(df, min_emp=50000, max_emp=300000)


# Function Returning Multiple Values [PY2]
def get_revenue_range(dataframe: pd.DataFrame):
    """
    [PY2]: Returns the min and max revenue from the dataframe.
    """
    min_revenue = dataframe['Revenue'].min()
    max_revenue = dataframe['Revenue'].max()
    return min_revenue, max_revenue


rev_min, rev_max = get_revenue_range(df)

# Sidebar Navigation [ST4]
st.sidebar.image("logo.png", use_container_width=True)

# Define pages and name them
page = st.sidebar.radio(
    "Navigate to:",
    ["Home", "Data Exploration", "Visualization & Analytics", "Mapping", "Additional Insights"]
)


# Home Page
def home_page():
    st.title("Fortune 500 Headquarters Explorer")
    st.write(
        "Welcome to the Fortune 500 Explorer! Discover details about Fortune 500 companies, including headquarters, revenue, profit, and more.")
    st.write("### How to Use:")
    st.write("- Use the sidebar to navigate between pages.")
    st.write("- Explore data, analytics, and interactive maps.")
    st.write("### Quick Facts:")
    st.write(f"- Total Companies: {len(df)}")
    st.write(f"- States Represented: {len(df['State'].unique())}")
    st.write(f"- Revenue Range: \u0024{rev_min}M - \u0024{rev_max}M")
    st.write("Start exploring the innovation hub!")


# Data Exploration Page
def data_exploration_page():
    st.title("Data Exploration")
    st.write("Use the filters below to explore and display the data.")

    # [ST1]: Dropdown for State selection
    all_states = sorted(df['State'].unique())
    selected_state = st.selectbox("Select a State:", options=["All"] + all_states)

    # [ST2]: Slider for minimum Revenue
    min_revenue_value = int(df['Revenue'].min())
    max_revenue_value = int(df['Revenue'].max())
    chosen_min_revenue = st.slider(
        "Minimum Revenue (millions):",
        min_value=min_revenue_value,
        max_value=max_revenue_value,
        value=min_revenue_value
    )

    # [ST3]: Multiselect for CompanyName
    all_companies = sorted(df['CompanyName'].unique())
    selected_companies = st.multiselect("Select Companies:", all_companies)

    # [DA4]: Filter by one condition (State) if not "All"
    if selected_state != "All":
        filtered_df = df[df['State'] == selected_state]
    else:
        filtered_df = df.copy()

    # [DA5]: Filter by multiple conditions (Revenue and Companies)
    filtered_df = filtered_df[filtered_df['Revenue'] >= chosen_min_revenue]

    # Sort by selected companies, if selected
    if len(selected_companies) > 0:
        filtered_df = filtered_df[filtered_df['CompanyName'].isin(selected_companies)]

    st.write("### Filtered Data")
    # [VIZ1]: Display a table of the filtered data
    # Selecting a filtered dataframe and highlighting
    # the max values of the 'Employees' and 'Revenue'
    # columns among all companies in light green
    st.dataframe(filtered_df.style.highlight_max(
        subset=['Revenue', 'Employees'],
        color='lightgreen'
    ))

    st.write("### Dictionary Example:")
    # Display dictionary (State and Count)
    for state_name, count_value in state_counts.items():
        st.write(f"State: {state_name}, Count: {count_value}")


# Visualization & Analytics Page
def visualization_analytics_page():
    st.title("Visualization & Analytics")

    # [DA3]: Top 10 by Employees
    top_employees = df.sort_values(by='Employees', ascending=False).head(10)

    st.write("### Top 10 Companies by Employees")
    # [VIZ2]: Bar Chart
    fig_bar = px.bar(
        top_employees,
        x='Employees',
        y='CompanyName',
        orientation='h',
        title="Top 10 Companies by Number of Employees",
        labels={'Employees': 'Employees', 'CompanyName': 'Company Name'},
        hover_data=['Revenue', 'Profit'],
        color='Employees',
        color_continuous_scale='Viridis'
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("### Scatter Plot: Revenue vs Profit")
    # [VIZ3]: Scatter plot of Revenue vs Profit
    fig_scatter = px.scatter(
        df,
        x='Revenue',
        y='Profit',
        color='State',
        title="Revenue vs Profit by State",
        labels={'Revenue': 'Revenue (millions)', 'Profit': 'Profit (millions)'},
        hover_data=['CompanyName', 'City'],
        size='Employees',
        size_max=15,
        opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("### Pivot Table: Average Revenue by State")
    # [DA6], [DA2]: Pivot table and sort
    pivot_table = df.pivot_table(
        values='Revenue',
        index='State',
        aggfunc='mean'
    )
    # Sort in descending order by average 'Revenue' by 'State'
    pivot_table = pivot_table.sort_values('Revenue', ascending=False)
    pivot_table = pivot_table.round(1)
    st.write(pivot_table)

    st.write("### Correlation Heatmap")
    # [VIZ4]: Correlation heatmap
    numeric_cols = [
        'Employees', 'Revenue', 'Profit',
        'RevenueBillions', 'ProfitMargin',
        'ProfitPerEmployee', 'RevenuePerEmployee', 'EfficiencyRatio'
    ]
    # Calculate correlation among numeric columns
    corr_matrix = df[numeric_cols].corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Correlation Heatmap of Numerical Features"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.write("### Treemap: Revenue Distribution by State")
    # [VIZ5]: Treemap
    # Group data by State and CompanyName, summing Revenue and Profit
    revenue_by_state = df.groupby(['State', 'CompanyName'])[['Revenue', 'Profit']].sum().reset_index()
    fig_treemap = px.treemap(
        revenue_by_state,
        path=['State', 'CompanyName'],
        values='Revenue',
        color='Revenue',
        color_continuous_scale='RdBu',
        title="Revenue Distribution by State and Company",
        hover_data=['Profit']
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    st.write("### Histogram: Profit Margin Distribution")
    # [VIZ6]: Histogram
    fig_hist = px.histogram(
        df,
        x='ProfitMargin',
        nbins=50,
        title="Distribution of Profit Margins",
        labels={'ProfitMargin': 'Profit Margin (%)'},
        color_discrete_sequence=['indianred']
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.write("### Top 10 Companies by Profit Per Employee")
    # [VIZ7]: Bar chart for Profit Per Employee
    # Creating a data frame with selected columns
    # and sorting by 'ProfitPerEmployee', displaying the first 10 rows
    profit_per_employee_df = df[['CompanyName', 'ProfitPerEmployee']].sort_values(
        by='ProfitPerEmployee',
        ascending=False
    ).head(10)
    fig_profit_emp = px.bar(
        profit_per_employee_df,
        x='ProfitPerEmployee',
        y='CompanyName',
        orientation='h',
        title="Top 10 Companies by Profit Per Employee",
        labels={'ProfitPerEmployee': 'Profit Per Employee (M)', 'CompanyName': 'Company Name'},
        color='ProfitPerEmployee',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_profit_emp, use_container_width=True)

    st.write("### Revenue vs Profit Efficiency Ratio")
    # Creating a data frame with selected columns
    # and sorting by 'EfficiencyRatio', displaying the first 10 rows
    efficiency_data = df[['CompanyName', 'EfficiencyRatio', 'Revenue', 'Profit']]
    efficiency_data = efficiency_data.sort_values(by='EfficiencyRatio', ascending=False).head(10)

    fig_efficiency = px.scatter(
        efficiency_data,
        x='Revenue',
        y='Profit',
        size='EfficiencyRatio',
        color='EfficiencyRatio',
        hover_name='CompanyName',
        title="Top 10 Companies by Efficiency Ratio (Profit/Revenue)",
        labels={
            'Revenue': 'Revenue (millions)',
            'Profit': 'Profit (millions)',
            'EfficiencyRatio': 'Efficiency Ratio'
        },
        size_max=15,
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig_efficiency, use_container_width=True)


# Mapping Page [MAP]
def mapping_page():
    # https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart
    st.title("Mapping: Headquarters Locations")
    st.write("Below is an interactive map showing the company headquarters.")

    # [DA9]: Additional calculations for marker radius
    # Using a function to determine radius from Employees
    def calculate_marker_radius(num_employees):
        # Use a formula for radius (r = √num_employees / 2)
        scale_factor = 5
        radius = np.sqrt(num_employees) * scale_factor
        return radius

    # [DA8]: Iterate through rows with iterrows()
    marker_radius = []
    for index, row in df.iterrows():
        radius = calculate_marker_radius(row['Employees'])
        marker_radius.append(radius)
    df['MarkerRadius'] = marker_radius

    # Dropdown for city selection to filter map
    unique_cities = sorted(df['City'].unique())
    selected_cities = st.multiselect(
        "Select Cities to Zoom and Filter:",
        options=unique_cities,
        default=None
    )

    # Zoom in if cities are selected
    if selected_cities:
        filtered_map_df = df[df['City'].isin(selected_cities)]
        avg_lat = filtered_map_df['Latitude'].mean()
        avg_lon = filtered_map_df['Longitude'].mean()
        zoom_level = 10
    # Show full map if no cities are selected
    else:
        filtered_map_df = df
        avg_lat = df['Latitude'].mean()
        avg_lon = df['Longitude'].mean()
        zoom_level = 3

    # https://deckgl.readthedocs.io/en/latest/view_state.html
    view_state = pdk.ViewState(
        latitude=avg_lat,
        longitude=avg_lon,
        zoom=zoom_level,
        pitch=0
    )

    # https://deckgl.readthedocs.io/en/latest/layer.html
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=filtered_map_df,
        get_position='[Longitude, Latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius='MarkerRadius',
        pickable=True
    )

    tooltip = {
        "html": "<b>Company:</b> {CompanyName} <br/><b>City:</b> {City}<br/><b>Revenue:</b> {Revenue} M<br/><b>Profit:</b> {Profit} M",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    )

    st.pydeck_chart(deck)


# Additional Insights Page
def additional_insights_page():
    st.title("Additional Insights & Advanced Analytics")

    # Existing Advanced Pivot Table with multiple aggregations
    multi_pivot = df.pivot_table(
        values=['Revenue', 'Profit'],
        index='State',
        aggfunc={
            'Revenue': ['mean', 'max'],
            'Profit': ['mean', 'min', 'max']
        }
    )
    st.write("### Multi-Aggregation Pivot Table (Revenue & Profit by State)")
    st.dataframe(multi_pivot)

    # Box Plot for Revenue distribution across states
    st.write("### Revenue Distribution Across States (Box Plot)")
    fig_box = px.box(
        df,
        x='State',
        y='Revenue',
        title="Box Plot of Revenue Distribution by State",
        labels={'Revenue': 'Revenue (millions)', 'State': 'State'},
        color='State',
        points="all"
    )
    fig_box.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig_box, use_container_width=True)

    # Error handling with user input for revenue
    revenue_input = st.text_input("Enter a minimum revenue (integer):", "10000")
    try:
        rev_val = int(revenue_input)
        filtered_rev_df = df[df['Revenue'] > rev_val]
        st.write(f"Companies with revenue > {rev_val}: {len(filtered_rev_df)}")
    except ValueError:
        st.error("Please enter a valid integer for revenue.")


# Main Routing
def main():
    if page == "Home":
        home_page()
    elif page == "Data Exploration":
        data_exploration_page()
    elif page == "Visualization & Analytics":
        visualization_analytics_page()
    elif page == "Mapping":
        mapping_page()
    elif page == "Additional Insights":
        additional_insights_page()


if __name__ == "__main__":
    main()