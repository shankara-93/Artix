# streamlit imports
import streamlit as st
from streamlit_extras.row import row
from streamlit_extras.metric_cards import style_metric_cards
import leafmap.foliumap as leafmap 


# python imports 
import os 
import cv2
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import plotly.graph_objs as go
from ultralytics import YOLO
import geopandas as gpd 
from geopy.geocoders import Nominatim
import time
from shapely.geometry import MultiPoint
from collections import OrderedDict 


#custom imports 
from backend.image_file_handlear import extract_dates_and_filenames_from_folder
from backend.utilities import get_max_date_from_dict,get_previous_dates_and_closest
from backend.model_infrances import run_inference
from backend.image_metadata import load_jpg_metadata,CoordinateTransformer
from backend.pixeltogeojson import boundingboxestogeodataframe 
from backend.parking_poly_utils import download_parking_data,intersecting_geojson 


def set_page_config():
    st.set_page_config(
        page_title="Parking Demo App",
        page_icon="🅿️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "https://docs.predicthq.com",
            "About": """
                **Parking Demo App**""",},)
# Set page config
set_page_config()


################################ data extraction Code #######################
# load yolo model
@st.cache_resource 
def load_model(path): 
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None 


# Get the absolute path of the current script (main_GPT.py)
current_dir = os.path.dirname(os.path.abspath(__file__)) 

st.write(current_dir)


# Navigate to the backend model directory dynamically
model_path = os.path.join(current_dir, "backend", "model", "best.pt") 

# Normalize the path (handles different OS path formats)
model_path = os.path.normpath(model_path)

#Load model from this location 
model= load_model(model_path) 
    
def boundingboxlist(_model, img_path): 
    try:
        bounding_boxes = run_inference(_model, img_path)
        return bounding_boxes 
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

def object_dection_date(date_file_dict):
    vehicle_bbox={}
    for key,item in date_file_dict.items():
        bbox = boundingboxlist(model, item) 
        vehicle_bbox[key]= bbox
    return vehicle_bbox 

transformer = CoordinateTransformer() 

def image_meta_data(date_file_dict_tif):
    image_meta={}
    for key,item in date_file_dict_tif.items():
        # Example usage
        metadata = load_jpg_metadata(item) 
        bbox_info = transformer.calculate_bbox_and_center(metadata['top_left_x'],metadata['top_left_y'],metadata['width_px'], metadata['height_px'],metadata['pixel_size'], from_crs=metadata['crs'], to_crs="EPSG:4326")
        image_meta[key]= {"metadata":metadata,"bbox_info":bbox_info} 
    return image_meta 

def bbox_geojson(vehicle_bbox_dict,metadata_dict):
    geojson_dict= {}
    for key, item in vehicle_bbox_dict.items():
        gdf_centroids = boundingboxestogeodataframe(vehicle_bbox_dict[key],metadata_dict[key]['metadata']['width_px'],metadata_dict[key]['metadata']['height_px'],metadata_dict[key]['bbox_info']['bounding_box'])
        gdf_parking =   download_parking_data(metadata_dict[key]['bbox_info']['bounding_box']) 
        gdf_vehicles_parking,parking_meta_data= intersecting_geojson(gdf_centroids,gdf_parking) 
        geojson_dict[key] = {'geojson_data_vehicle': gdf_centroids,"geojson_parking":gdf_parking,"geojson_intersection":gdf_vehicles_parking,"parking_meta_data":parking_meta_data} 
    return geojson_dict


@st.cache_data 
def list_images(tab,base_path):
    sel_path= os.path.join(base_path,tab)
    date_list, date_file_dict = extract_dates_and_filenames_from_folder(sel_path,file_type=".jpg") 
    #__, date_file_dict_tif= extract_dates_and_filenames_from_folder(sel_path,file_type=".tif") 
    max_date = get_max_date_from_dict(date_file_dict)
    vehicle_bbox= object_dection_date(date_file_dict)
    images_metadata= image_meta_data(date_file_dict) 
    geojson_dict= bbox_geojson(vehicle_bbox,images_metadata) 
    
    return date_list, date_file_dict,max_date,images_metadata,geojson_dict 

def data_picker_date(date,geojson_dict):
    table_df= geojson_dict[date]['geojson_intersection'] 
    table_df= table_df[["osmid","Category","occupancy","Capacity","area_m2"]] 
    return table_df 

def previous_date_cal(date_dict,current_date,geojson_dict):
    previous_dates, closest_previous_key= get_previous_dates_and_closest(date_dict,current_date)
    previous_dates_dict={}
    for date in previous_dates:
        data_df=data_picker_date(date,geojson_dict) 
        previous_dates_dict[date]={"Parking_vol":int(data_df['occupancy'].sum()),"parking_area":data_df['area_m2'].sum().round(2)}
    return previous_dates_dict,closest_previous_key  

@st.cache_data 
def get_city_name(bounding_box):
    min_lat, min_lon, max_lat, max_lon =  bounding_box
    lat,lon=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2] 
 
    geolocator = Nominatim(user_agent="geoapiExercises")
    time.sleep(3)
    location = geolocator.reverse((lat, lon), language='en', exactly_one=True)
    
    if location and location.raw.get('address', {}).get('city'):
        return location.raw['address']['city']
    elif location and location.raw.get('address', {}).get('town'):
        return location.raw['address']['town']
    elif location and location.raw.get('address', {}).get('village'):
        return location.raw['address']['village']
    else:
        return "City not found"


###################### UI code #########################################

#### 
def plot_area_chart_plotly(df):
    fig = go.Figure()

    # Add trace for Apple data as an area chart
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Parking_vol'],
        mode='lines',
        fill='tozeroy',  # Fills the area under the line
        line=dict(color='green'))) 

    # Set layout to control height
    fig.update_layout(
        xaxis_title='Timeline',
        yaxis_title='Occupancy',
        height=250  # Set the plot height to fit within the container
    )

    # Plot using Plotly
    st.plotly_chart(fig, use_container_width=True)

    
def multipoint_to_dataframe(geo_dataframe):
    """
    Converts a GeoDataFrame containing MultiPoint or Point geometries into a pandas DataFrame 
    with columns for longitude, latitude, and a weight column indicating the number of points
    in each MultiPoint geometry.
    
    Parameters:
    geo_dataframe (GeoDataFrame): A GeoDataFrame with geometry column containing MultiPoint or Point geometries.
    
    Returns:
    pd.DataFrame: A DataFrame containing the longitude, latitude, and weight for all points.
    """
    locations_road = []
    
    for geom in geo_dataframe.geometry:
        if geom.geom_type == 'MultiPoint':
            num_points = len(geom.geoms)  # Get the number of points in the MultiPoint
            for point in geom.geoms:
                locations_road.append([point.x, point.y, num_points])  # Append with weight (number of points)
        elif geom.geom_type == 'Point':  # Handle single Point geometries
            locations_road.append([geom.x, geom.y, 1])  # Single points get a weight of 1
    
    # Convert the list to a pandas DataFrame
    df_road = pd.DataFrame(locations_road, columns=["longitude", "latitude", "weight"])
    
    return df_road

    
def map_plot(data,bounding_box, zoom= 15,basemap="CartoDB.Positron"):
    min_lat, min_lon, max_lat, max_lon =  bounding_box
    center_lat_long=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2] 
    
    # Step 1: Prepare the polygons for visualization
    polygons_layer = data[['polygon_geometry']]
    polygons_layer = polygons_layer.set_geometry('polygon_geometry')

    # Step 2: Prepare the points for visualization
    points_layer = data[['geometry']]
    points_layer = points_layer.set_geometry('geometry') 
    df_road = multipoint_to_dataframe(points_layer) 
    m = leafmap.Map(center=center_lat_long, zoom=zoom, height="300px", width="800px",widescreen=False) 
    m.add_basemap(basemap) 
    m.add_gdf(polygons_layer, layer_name="parking_area",fill_colors=["red"]) 
    m.add_circle_markers_from_xy(df_road,x="longitude", y="latitude", radius=3, color="blue", fill_color="blue",layer_name="cars_road")
    m.add_text("Artix EngineX", position="bottomright")
    #m.add_image(image_path,bounding_box) 
    m.to_streamlit(use_container_width=False)

def heat_map(data, bounding_box, zoom= 15,basemap="CartoDB.Positron"):
    min_lat, min_lon, max_lat, max_lon =  bounding_box
    center_lat_long=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2] 
    
    points_layer = data[['geometry']]
    points_layer = points_layer.set_geometry('geometry') 
    df_road = multipoint_to_dataframe(points_layer)
    
    m = leafmap.Map(center=center_lat_long, zoom=zoom,height="400px", width="800px", widescreen=False)
    m.add_basemap(basemap) 
    m.add_heatmap(df_road,latitude="latitude",longitude="longitude",value="weight",name="Heat map",radius=10,)
    colors = ["blue", "lime", "red"]
    vmin = df_road['weight'].min()
    vmax = df_road['weight'].max()

    m.add_colorbar(colors=colors, vmin=vmin, vmax=vmax)
    m.add_text("Artix EngineX", position="bottomright")
    m.to_streamlit()
    
    
def main_dashborad_UI(res): 
    st.markdown(f'<span style="color:#4F8A8B"><b>Selected Resolution: {res}</b></span>', unsafe_allow_html=True) 
    __, date_file_dict,max_date,images_metadata,geojson_dict= list_images(res,base_path) 
    with st.expander("Pick Image Date"):
        selected_date = st.sidebar.selectbox("**Pick a Date:**",list(date_file_dict.keys()),index=list(date_file_dict.keys()).index(max_date)) 
        
    ## data based on selected_date 
    current_df= data_picker_date(selected_date,geojson_dict) 
    parking_intersection_df=geojson_dict[selected_date]['geojson_intersection'] 
    parking_place= geojson_dict[selected_date]['parking_meta_data'] 
    boundingbox_map= images_metadata[selected_date]['bbox_info']['bounding_box'] 
    #image_path = date_file_dict[selected_date] 
    #city = get_city_name(boundingbox_map) 
    city= "Bunnik"
    current_Occupied= current_df['occupancy'].sum() 
    current_occupied_area= current_Occupied*12.50
    avg_parking_current_date= current_df['occupancy'].mean().round(1)

    ## previous_dates 
    previous_dict_data,closest_previous_key= previous_date_cal(date_file_dict,selected_date,geojson_dict) 
    previous_data_df= pd.DataFrame.from_dict(OrderedDict(sorted(previous_dict_data.items())), orient='index').reset_index()
    previous_data_df= previous_data_df.rename(columns={'index': 'Date'})
    previous_data_df['Date'] = pd.to_datetime(previous_data_df['Date'], format='%Y-%m-%d_%H%M%S') 
    previous_date= closest_previous_key.split("_")[0]
    date_obj = datetime.strptime(previous_date, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%d-%m-%Y")
    
    #st.sidebar.write(previous_data_df) 
    
    def calculate_percentage_change(current_val, previous_val):
        """
        Calculate the percentage change between the current value and the previous value.

        :param current_val: The current value.
        :param previous_val: The previous value.
        :return: The percentage change rounded to 2 decimal places, or None if previous_val is zero.
        """
        if previous_val == 0:
            return None  # Handle division by zero
        
        percentage_change = ((current_val - previous_val) / previous_val) * 100
        return round(percentage_change, 2)
    
    previous_occupied= previous_dict_data[closest_previous_key]['Parking_vol'] 
    previous_occupied_area= previous_occupied*12.50
    
    if selected_date==closest_previous_key: 
        delta_occupancy= None
        delta_area=None
    else:
        delta_occupancy= calculate_percentage_change(current_Occupied,previous_occupied)
        delta_occupancy= f"{int(delta_occupancy)}% w.r.t ({formatted_date})" 
        delta_area= calculate_percentage_change(current_occupied_area,previous_occupied_area)
        delta_area= f"{delta_area}% w.r.t ({formatted_date})" 
    
        
    with st.container(height=300,border=True):
        col1, col2 = st.columns((0.30, 0.70))
        with col1.container(height=275, border=False):
            st.markdown("""<h5 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>City Parking Metadata</h5>""", unsafe_allow_html=True)
            st.markdown(" ")
            row1 = st.columns(2)
            row1[0].metric(label="City", value=f"{city}", delta=None)
            row1[1].metric(label="Num parking places", value=parking_place['num_parking'], delta=None)
            row2 = st.columns(2)
            row2[0].metric(label="Parking Capacity", value=parking_place['parking_capacity'], delta=None)
            row2[1].metric(label="Parking Area (m2)", value=f"{int(parking_place['parking_area_capacity'])}", delta=None)
            style_metric_cards(border_radius_px=10)

        with col2.container(height=275, border=False):
            st.markdown("""<h5 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Car Park Occupancy Graph</h5>""", unsafe_allow_html=True)
            st.markdown(" ")
            row1, row2 = st.columns((0.75, 0.25))
            with row1:
                plot_area_chart_plotly(previous_data_df)
            with row2:
                st.metric(label="Total Vehicles", value=previous_data_df['Parking_vol'].sum())
                st.metric(label="Min Vehicles", value=previous_data_df['Parking_vol'].min())
                style_metric_cards(border_radius_px=10)
                     
    with st.container(height=400,border=True):
        col1, col2= st.columns((0.60, 0.40))
        with col1.container(height=375, border=False):
            st.markdown("""<h4 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Occupancy City Map</h4>""", unsafe_allow_html=True)
            st.markdown(" ")
            map_plot(parking_intersection_df,boundingbox_map) 
        with col2.container(height=375, border=False):
            st.markdown("""<h4 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Parking Events</h4>""", unsafe_allow_html=True)
            st.markdown(" ")
            #st.tabs(["Day", "Week", "Month"])
            row1 = row(2, vertical_align="top")
            row1.metric(label="Total Occupied", value=current_Occupied, delta= delta_occupancy) 
            row1.metric(label="Parking Capacity (m2)", value=int(previous_occupied_area), delta=delta_area)
            row2 = row(2, vertical_align="center")
            row2.metric(label="Avg Occupancy", value=int(avg_parking_current_date), delta=None) # need to add Avg parking places 
            #row2.metric(label="City", value=5000, delta=1000)
            style_metric_cards(border_radius_px=10)  
    with st.container(height=400,border=True):
        col1, col2= st.columns((0.40, 0.60))
        with col1.container(height=375, border=False):
            st.markdown("""<h4 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Data Insights</h4>""", unsafe_allow_html=True)
            st.markdown(" ")
            #st.markdown("**Data Insights**") ## (Utilization)
            st.table(current_df)
        with col2.container(height=375, border=False): 
            st.markdown("""<h4 style='color: #4F8A8B; background-color: #E0F7FA; padding: 10px; border-radius: 8px;'>Occupancy Heat Map</h4>""", unsafe_allow_html=True)
            st.markdown(" ")
            heat_map(parking_intersection_df,boundingbox_map) 
    
    



st.sidebar.title(":green[Artix Technologies]")  
with st.expander(label="Select Image Resolution", expanded=True):
    res = st.sidebar.selectbox("**Select Resolution:**", ["30CM", "50CM", "80CM"],help="select satellite images resolution for analysis")
 
#Base path all images 

# Navigate up to the 'aws' folder
aws_folder = os.path.dirname(current_dir)

# Set the base path to the 'Data' folder inside 'aws'
base_path = os.path.join(aws_folder, "Data") 


if res=="30CM":
    main_dashborad_UI(res)
elif res=="50CM":
    main_dashborad_UI(res)
else:
    st.markdown(f"Selected Resolution model development is under process")

st.sidebar.title("About") 
logo = os.path.join(aws_folder, "logo",'Investor Pitch_SV.jpg') 
st.sidebar.image(logo)  
st.sidebar.title("Contact")

st.sidebar.info(
    """
    📍 **Address:** Oder 20, UNIT A1164, The Hague, 2491DC  
    📧 **Email:** info@artixtechnologies.com  
    📞 **Phone:** +31 0622385211  
    🏢 **KVK No:** 94868468
    """)