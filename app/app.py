# import folium
# import pandas as pd


# m = folium.Map(location=(49.25, -123.12), tiles="cartodb positron")

# location = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource/2dec86ed-76ed-47a3-ae28-646db5c5b965/download/dublin.csv"
# bike_station_locations = pd.read_csv(location)

# bike_station_locations = bike_station_locations[["Latitude", "Longitude", "Name"]]


# map = folium.Map(
#     location=[bike_station_locations.Latitude.mean(), bike_station_locations.Longitude.mean()],
#     zoom_start=14,
#     control_scale=True
# )

# for index, location_info in bike_station_locations.iterrows():
#     folium.Marker([location_info["Latitude"], location_info["Longitude"]], popup=location_info["Name"]).add_to(map)

# map
# map.save("footprint.html")


import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium


APP_TITLE = 'PIWO PIWO PIWO PIWO'
APP_SUB_TITLE = 'GOLEM GUROM'


def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)


if __name__ == "__main__":
    main()
