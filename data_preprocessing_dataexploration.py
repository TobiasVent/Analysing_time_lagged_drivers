import pandas as pd
import pickle
import os

def get_zoned_df(appended_data):
    """
    Split the dataframe into four different ocean basins
    based on latitude and longitude.
    """

    # Arctic region
    zone_ARCTIC = appended_data.loc[appended_data['nav_lat'] > 70.0]
    zone_ARCTIC['zone'] = 'ARCTIC'
        
    # North Atlantic region
    zone_NORTH_ATLANTIC = appended_data.loc[
        (appended_data['nav_lon'] >= -75.0) & (appended_data['nav_lon'] <= 0.0)
    ]
    zone_NORTH_ATLANTIC = zone_NORTH_ATLANTIC.loc[
        (zone_NORTH_ATLANTIC['nav_lat'] >= 10) &
        (zone_NORTH_ATLANTIC['nav_lat'] <= 70)
    ]
    zone_NORTH_ATLANTIC['zone'] = 'NORTH_ATLANTIC'
    
    # Equatorial Pacific region
    zone_EQ = appended_data.loc[
        (appended_data['nav_lat'] >= -10.0) &
        (appended_data['nav_lat'] <= 10.0)
    ]
    zone_EQ_PACIFIC_1 = zone_EQ.loc[
        (zone_EQ['nav_lon'] >= 105.0) & (zone_EQ['nav_lon'] <= 180.0)
    ]
    zone_EQ_PACIFIC_2 = zone_EQ.loc[
        (zone_EQ['nav_lon'] >= -180.0) & (zone_EQ['nav_lon'] <= -80.0)
    ]
    zone_EQ_PACIFIC = pd.concat([zone_EQ_PACIFIC_1, zone_EQ_PACIFIC_2])
    zone_EQ_PACIFIC['zone'] = 'EQ_PACIFIC'
    
    # Southern Ocean region
    zone_SOUTHERN_OCEAN = appended_data.loc[appended_data['nav_lat'] <= -45]
    zone_SOUTHERN_OCEAN['zone'] = 'SOUTHERN_OCEAN'
    
    return zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN

time_series = pd.DataFrame()
zone_NORTH_ATLANTIC_SERIES = pd.DataFrame()
zone_SOUTHERN_OCEAN = pd.DataFrame()





def concat_data(range_start,range_end,frac,experiment_name,region):




    for i in range(range_start,range_end+1):
        print(i)
        if experiment_name == "experiment_1":
            file_path = f"/data/experiment_data/experiment_1/1/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.hind_{i}_df.pkl"
        if experiment_name == "experiment_5":
            file_path = f"/data/experiment_data/experiment_5/ORCA025.L46.LIM2vp.CFCSF6.MOPS.JRA.LP04-KLP002.wind_{i}_df.pkl"
        df = pd.read_pickle(file_path)
        df = df[df["tmask"]==1]
        df = df.drop(columns= ['tmask','y','x','time_centered','e1t','e2t'])
        df_coords = df[['nav_lat','nav_lon']].drop_duplicates().reset_index(drop=True)

        df_coords["coord_id"] = df_coords.index
        df_coords_sampled = df_coords.sample(frac=frac, random_state=42)
        df = df.merge(df_coords_sampled, on=['nav_lat', 'nav_lon'], how='inner')
        df = df.sort_values(by= ["coord_id", "time_counter"])

        if region == 'Southern_Ocean':
            zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN = get_zoned_df(df)
            time_series.append(zone_SOUTHERN_OCEAN)

        if region == 'North_Atlantic':
            zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN = get_zoned_df(df)
            time_series.append(zone_NORTH_ATLANTIC)
        else:
            print("Region not defined properly. Please choose 'global', 'Southern_Ocean' or 'North_Atlantic'")
        time_series = pd.concat([time_series,df])

    out_dir = "data/data_exploration/concatenated_years"
    os.makedirs(out_dir, exist_ok=True)

    out_file = f"{out_dir}/{region}_{range_start}_{range_end}_{experiment_name}.pkl"

    with open(out_file, "wb") as f:
        pickle.dump(time_series, f)



concat_data(1958,2018,1,"expirment_1","North_Atlantic")
concat_data(1958,2018,1,"expirment_1","Southern_Ocean")
concat_data(1958,2018,1,"expirment_5","North_Atlantic")
concat_data(1958,2018,1,"expirment_5","Southern_Ocean")


    

    

