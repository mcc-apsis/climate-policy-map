# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ********** LOADING **********

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import pandas as pd
import geopy
import geopy.distance
import numpy as np
import shapely.vectorized

# %% [markdown]
# ********** FUNCTIONS FOR CREATING the DENSITY GRID **********

# %%
#Max's
def new_haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth.

    Args:
        lon1 (float): Longitude of point 1 (specified in decimal degrees).
        lat1 (float): Latitude of point 1 (specified in decimal degrees).
        lon2 (float): Longitude of point 2 (specified in decimal degrees).
        lat2 (float): Latitude of point 2 (specified in decimal degrees).

    Returns:
        Distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1[:,None]
    dlat = lat2 - lat1[:,None]

    a = np.sin(dlat/2.0)**2 + np.cos(lat1[:,None]) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    
    return km


# %%
# Max's
def density_grid(degrees,distance,df):
    df_countries = df[df["feature_code"]=="PCLI"]
    df_places = df[df["feature_code"]!="PCLI"]

    # linspace(start, stop, number of evenly spaced numbers)
    latbins = np.linspace(-90,90, round(180/degrees))
    lonbins = np.linspace(-180,180, round(360/degrees))

    n = np.zeros((len(latbins),len(lonbins)))
    
    print(f"calculating density grid of size: {n.size}")

    for i,lat in enumerate(latbins):
        # Calculating the geodesic distance between two points
        r = geopy.distance.distance(kilometers=distance)
        latp = geopy.Point((lat,135))
        # if the latitude is closer than distance to the north pole, then the northern bound should be 
        # the north pole, not distancekm north of the latitude (which will pass the pole and go south again)
        if geopy.distance.great_circle(latp,(90,135)).km < distance:
            r_nbound = 90   
        else:
            r_nbound = r.destination(point=latp,bearing=0).latitude
        # Same as above for the south pole
        if geopy.distance.great_circle(latp,(-90,135)).km < distance:
            r_sbound = -90   
        else:
            r_sbound = r.destination(point=latp,bearing=180).latitude        

        latbound_df = df_places[
            (df_places.lat>=r_sbound) &
            (df_places.lat<=r_nbound)        
        ]

        ds = new_haversine_np(latbound_df['lon'], latbound_df['lat'],lonbins,[lat]*len(lonbins))

        n[i,:] = np.where(ds<distance,1,0).sum(axis=0)
        
    print("done")
    
    shpfilename = shpreader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    yv, xv = np.meshgrid(latbins, lonbins)

    for country in reader.records():
        incountry = shapely.vectorized.contains(country.geometry,xv,yv)
        idx = np.argwhere(incountry==True)
        ndots = idx.size/2
        cdf = df_countries[df_countries["country_predicted"]==country.attributes["SU_A3"]]
        for point in idx:
            n[point[1],point[0]] += cdf.shape[0]/ndots

    return latbins, lonbins, n

# %% [markdown]
# ********** DATA MANIPULATION **********

# %%
def create_list_of_predicted_categories(df):
    """Creates a list with all categories in the dataframe and a list with categories of the dataframe that have predictions.

    Args:
        df (DataFrame): The input dataframe (combined document ratings and geo data).

    Returns:
        all_cats: A list with all categories in the dataframe.
        pred_cats: A list with categories that have predictions.
    """
    data = df.copy()
    
    all_cats = [c for c  in data.columns if ("-" in c) and ("prediction" not in c) and ("hidden" not in c)] # This list contains all variable columns.
    preds = [c for c in data.columns if "prediction" in c and "hidden" not in c] # This list contains all columns that are predictions.
    pred_cats = []
    for c in all_cats:
        for p in preds:
            if c in p:
                pred_cats.append(c)
    return all_cats, pred_cats


# %%
def merge_columns(df, col, col_pred):
    """Creates a dataframe with a merged human/prediction column.

    Args:
        df (DataFrame): The input dataframe (combined document ratings and geo data).
        col (str): Name of the column that contains the variable.
        col_pred (str): Name of the column with the predicted variable.

    Returns:
        A modified dataframe with a column that has the predicted values if the paper was not seen by a human, and the "human" values if the paper was seen. The column has the same name of the variable.
    """
    
    data = df.copy()
    data['human'] = data[col] # This creates a copy of human data under the "human" header
    data[col] = data[col_pred] # This replaces the column of interest (human data) with predicted data
    
    i = 0
    while(i < len(data)):
        if data['seen'].iloc[i] == 1:
            data.at[i, col] = data['human'].iloc[i] # This replaces the predicted data, which is under the "column of interest" header, with the human data from the copied column.
        i = i + 1

    return data


# %%
def filter(df, cat, prob=0.5):
    """Filters a dataframe based on a variable of interest.

    Args:
        df (DataFrame): The input dataframe (combined document ratings and geo data).
        cat (str): Name of the column that contains the variable. 'All' if no filtering is to be done.
        prob (float): Minimum probability threshold to consider that a paper is about the variable of interest. Default: 0.5

    Returns:
        A dataframe that has been filtered according to a specific variable.
    """
    filtered = df.copy()
    all_cats, pred_cats = create_list_of_predicted_categories(df)
    
    if(cat == 'All' or cat == 'all'):
        return filtered
    elif(cat in pred_cats):
        pred = cat + ' - prediction'
        filtered = merge_columns(filtered, cat, pred)
        filtered = filtered[filtered[cat] >= prob]
        return filtered
    elif(cat in all_cats):
        filtered = filtered[filtered[cat] >= prob]
        return filtered
    else:
        print('Wrong variable')


# %%
def filter_by_country(df, country_code, country_col='country_predicted'):
    """ Creates a dataframe of articles about the specified country.

        Args:
            df (DataFrame): Dataframe with article data.
            country_code (str): World Banck country code of the country of interest.
            country_col (str): Name of the column with the countries. Default: 'country_predicted'

        Returns:
            Dataframe with articles about the specified country.

    """    
    data = df[df[country_col]==country_code].sort_values(by=['id']).reset_index(drop=True)
    return data


# %%
def remove_mislabeled_rows(df, country_code, associations, country_col='country_predicted'):
    """
        Remove rows that correspond to most probably mislocalised articles due to specific expressions, like "Paris Agreement".

        Args:
            df (DataFrame): Dataframe to correct for possible mislocalisations.
            country_code (str): Country code as used by the World Bank.
            associations (list): List of expressions that may generate mislabeling.
            country_col (str): Column with country codes.

        Returns:
            Dataframe without rows that most possibly are mislabeled.
    """
    # Country-exclusive data    
    dat_country = filter_by_country(df, country_code, country_col)
    # Removing country data from the total
    data = df[df[country_col]!=country_code]
    
    inds_to_drop = []
    for exp in associations:
        for i in range(0, len(dat_country)):
            title = dat_country.iloc[i]['title']
            if exp in title:
                inds_to_drop.append(i)
            else:
                abstract = dat_country.iloc[i]['content']
                if exp in abstract:
                    inds_to_drop.append(i)
    dat_country = dat_country.drop(index=inds_to_drop)
    data = pd.concat([data, dat_country])
    data = data.sort_values(by=['id']).reset_index(drop=True)
    return data    


# %%
def create_mismatch_dictionary(df, df2, country_column='aff_country'):
    """
       Creates a dictionary of correct matches between countries in a list and World Bank country codes.

       Args:
        df (DataFrame): Dataframe with a column of countries.
        df2 (DataFrame): Country classification table (data/country_classification.xls')
        country_column (str): Name of column with country names.
    """
    mismatches = []
    for country in df[country_column]:
        if country not in list(df2['Economy']):
           mismatches.append(country)
    mismatches = pd.unique(mismatches)
    # Manually establishing pairs
    mismatch_pairs = {}
    for mismatch in mismatches:
        print('Introduce country code for '+mismatch+':')
        code = input()
        mismatch_pairs[mismatch] = code
    # Adding all pairs
    i = 0
    while i < len(df2):
        country = df2['Economy'][i]
        code = df2['Code'][i]
        mismatch_pairs[country] = code
        i = i+1

    f = open("../data/complete_country_codes.txt","w")
    f.write( str(mismatch_pairs) )
    f.close()


# %%
def merge_human_and_predicted(df, cols):
    """
    Merges the human and predicted columns of a variable.

    Args:
        df (DataFrame): DataFrame on which to merge human and predicted columns.
        cols (list): List of columns that have an analogous " - predicted" column. See create_list_of_predicted_categories(data).

    Return:
        A DataFrame where the human column of each variable is now the human column, unless it was not seen, in which case it would be the predicted value.
    """
    data = df.copy()
    for col in cols:
        data = merge_columns(data, col, col+' - prediction')
    return data


# %%
def normalise_by_col_and_row(df):
    """
        Normalises a dataframe by row and, separately, by column.
        
        Args:
            df (DataFrame): Dataframe to normalise.
        
        Returns:
            Dataframe normalised by column. NAs filled with zeroes.
            Dataframe normalised by row. NAs filled with zeroes.
    """
    col_norm = df/df.sum(axis=0)
    row_norm = df.div(df.sum(axis=1), axis=0)
    col_norm.fillna(value=0, inplace=True)
    row_norm.fillna(value=0, inplace=True)
    return col_norm, row_norm

# %% [markdown]
# ********** PLOTTING **********

# %%
def plot_map(cat, degrees, distance, df, file_name, dpi=150, width=8, height=5):
    """Creates a map based on a variable of interest and saves it in a file.

    Args:
        cat (str): Variable of interest. "All" if all papers are to be mapped.
        degrees (float): Number of degrees.
        distance (int): Distance of the clusters.
        df (DataFrame): The input dataframe (combined document ratings and geo data).
        file_name (str): Name of the plot file.
        dpi (int): Resolution (dots per inch).
        width (int): Width of the plot. Default: 8.
        height (int): Height of the plot. Default: 5.
    """
    fig, ax = plt.subplots(dpi=dpi, figsize=(width, height))
    p = ccrs.Mollweide()
    ax = plt.axes(projection=p)
    ax.set_global()
    ax.coastlines(lw=0.1)

    filtered = filter(df, cat)
    latbins, lonbins, n = density_grid(degrees,distance,filtered)

    vm = n[~np.isnan(n)].max()
    n[n == 0] = np.nan

    pcm = plt.pcolormesh( 
        lonbins, latbins, n,
        transform=ccrs.PlateCarree(),
        norm=mpl.colors.LogNorm(vmin=1, vmax=vm),
        alpha=0.5,
        cmap="YlGnBu"
    )

    fig.colorbar(pcm)
    title = cat.split(" - ")[1]
    ax.set_title(title)
    filename = '../plots/density_maps/' + file_name + '.pdf'
    plt.savefig(filename)


# %%
def plot_maps(df):
    """Plots all variables in a dataframe.

    Args:
        df (DataFrame): The input dataframe (combined document ratings and geo data).
    """
    all_cats, pred_cats = create_list_of_predicted_categories(df)
    for counter, c in enumerate(all_cats):
        file_name = c.split(' - ')[1].replace(" ","_").replace("/", "_")
        print(file_name)
        plot_map(c, 1, 100, df, file_name)
        print(str((counter/len(all_cats))*100) + '%')


