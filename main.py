import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
import datetime
import tacotoolbox
import tacoreader
import ee

ee.Initialize()

# Main directory containing the dataset
dir = pathlib.Path("/media/disk/databases/WORLDFLOODS/2_Mart/hugging_face")
list_tiff = [str(file) for file in dir.glob("**/*.tif")]

# Prepare metadata DataFrame
df = pd.DataFrame(list_tiff, columns=["path"])
df["split"] = df["path"].str.split("/").str[7].replace("val", "validation")
df["name_file"] = df["path"].str.split("/").str[-1].str.split(".").str[0]
df["name_folder"] = df["path"].str.split("/").str[-2]
df["tortilla_path"] = "/data/databases/Julio/Flood_kike/tortillas/" + df["name_file"] + ".tortilla"

# Load base metadata
base_df = pd.read_csv("/media/disk/databases/WORLDFLOODS/2_Mart/hugging_face/dataset_metadata.csv")
base_df_modified = base_df.drop(columns=['split'])
base_df_modified.columns = [
    'event_id', 'source', 'ems_code', 'aoi_code', 'satellite',
    'satellite_date', 's2_date', 'bounds', 'crs', 'transform'
]

# Merge metadata with file paths
merged_df = df.merge(base_df_modified, left_on='name_file', right_on='event_id', how='left')
merged_df["satellite_dt"] = pd.to_datetime(merged_df["satellite_date"], format='ISO8601', errors='coerce')
merged_df["s2_dt"] = pd.to_datetime(merged_df["s2_date"], format='ISO8601', errors='coerce')
merged_df["date_diff_days"] = abs((merged_df["satellite_dt"] - merged_df["s2_dt"]).dt.days)

# Group by file name for processing
grouped = merged_df.groupby("name_file")


# Process each group and create tortilla files
for i, (file_name, group) in enumerate(grouped):
    if i % 10 == 0:
        print(f"Processing {i}/{len(grouped)}")

    row_s2 = group[group["name_folder"] == "S2"].iloc[0]
    row_gt = group[group["name_folder"] == "gt"].iloc[0]
    row_per = group[group["name_folder"] == "PERMANENTWATERJRC"].iloc[0]

    # Read Sentinel-2 data and create tortilla sample
    profile_row_s2 = rio.open(row_s2["path"]).profile
    sample_row_s2 = tacotoolbox.tortilla.datamodel.Sample(
        id=row_s2["name_folder"],
        path=row_s2["path"],
        file_format="GTiff",
        data_split=row_s2["split"],
        stac_data={
            "crs": "EPSG:" + str(profile_row_s2["crs"].to_epsg()),
            "geotransform": profile_row_s2["transform"].to_gdal(),
            "raster_shape": (profile_row_s2["height"], profile_row_s2["width"]),
            "time_start": datetime.datetime.fromisoformat(row_s2.s2_date),
            "time_end": datetime.datetime.fromisoformat(row_s2.s2_date),
        },
    )

    # Read Permanent Water data and create tortilla sample
    profile_row_per = rio.open(row_per["path"]).profile
    sample_row_per = tacotoolbox.tortilla.datamodel.Sample(
        id=row_per["name_folder"],
        path=row_per["path"],
        file_format="GTiff",
        data_split=row_per["split"],
        stac_data={
            "crs": "EPSG:" + str(profile_row_per["crs"].to_epsg()),
            "geotransform": profile_row_per["transform"].to_gdal(),
            "raster_shape": (profile_row_per["height"], profile_row_per["width"]),
            "time_start": datetime.datetime.fromisoformat(row_per.satellite_date),
            "time_end": datetime.datetime.fromisoformat(row_per.satellite_date),
        },
    )

    # Read Ground Truth data and create tortilla sample
    profile_row_gt = rio.open(row_gt["path"]).profile
    sample_row_gt = tacotoolbox.tortilla.datamodel.Sample(
        id=row_gt["name_folder"],
        path=row_gt["path"],
        file_format="GTiff",
        data_split=row_gt["split"],
        stac_data={
            "crs": "EPSG:" + str(profile_row_gt["crs"].to_epsg()),
            "geotransform": profile_row_gt["transform"].to_gdal(),
            "raster_shape": (profile_row_gt["height"], profile_row_gt["width"]),
            "time_start": datetime.datetime.fromisoformat(row_gt.satellite_date),
            "time_end": datetime.datetime.fromisoformat(row_gt.satellite_date),
        },
    )

    # Create Samples object
    samples = tacotoolbox.tortilla.datamodel.Samples(
        samples=[sample_row_s2, sample_row_per, sample_row_gt]
    )

    # Create tortilla file for the samples
    tacotoolbox.tortilla.create(samples, row_s2["tortilla_path"], quiet=True)

# Create a collection of tortilla samples
sample_tortillas = []
for i, (file_name, group) in enumerate(grouped, start=1):
    if i % 10 == 0:
        print(f"Processing {i}/{len(grouped)}")

    row = group[group["name_folder"] == "S2"].iloc[0]

    # Load tortilla data
    sample_data = tacoreader.load(row["tortilla_path"])
    sample_data = sample_data.iloc[0]

    # Create a sample for the tortilla data
    sample_tortilla = tacotoolbox.tortilla.datamodel.Sample(
        id=pathlib.Path(row["tortilla_path"]).stem,
        path=row["tortilla_path"],
        file_format="TORTILLA",
        data_split=row["split"],
        stac_data={
            "crs": sample_data["stac:crs"],
            "geotransform": sample_data["stac:geotransform"],
            "raster_shape": sample_data["stac:raster_shape"],
            "centroid": sample_data["stac:centroid"],
            "time_start": sample_data["stac:time_start"],
            "time_end": sample_data["stac:time_end"],
        },
        source=row["source"],
        ems_code=row["ems_code"],
        aoi_code=row["aoi_code"],
        satellite=row["satellite"],
        satellite_date=row["satellite_date"],
        s2_date=row["s2_date"],
        days_diff=row["date_diff_days"]
    )
    sample_tortillas.append(sample_tortilla)

# Create final collection and add metadata
samples_obj = tacotoolbox.tortilla.datamodel.Samples(samples=sample_tortillas)
samples_obj = samples_obj.include_rai_metadata(
    sample_footprint=5120, cache=False, quiet=False
)

# Set collection metadata and description
description = """

### Database

**WorldFloods** is a public dataset containing pairs of Sentinel-2 multispectral images (Level-1C) and corresponding flood segmentation masks. As of its latest release, it comprises **509** flood events worldwide, requiring approximately **300 GB** of storage if fully downloaded. The primary goal is to facilitate **automatic flood mapping** from optical satellite data by providing high-quality training and validation samples.

Each **image-mask** pair captures a satellite scene (with all 13 Sentinel-2 bands) and the corresponding **binary flood masks**, refined or curated from authoritative flood extent products by services such as [Copernicus EMS](https://emergency.copernicus.eu/) and [UNOSAT](https://unitar.org/maps). Flood events range from 2016 to 2023, encompassing diverse hydrological conditions and global regions.

### Sensors Used

- **Sentinel-2 MSI (Multispectral Instrument):**
  - 13 spectral bands spanning from visible (VNIR) to shortwave infrared (SWIR).
  - Spatial resolutions of 10 m, 20 m, or 60 m, which are harmonized to 10 m in the dataset for consistent pixel-size alignment.
  - Scenes are **Level-1C** or **Level-2A** reflectance products, further processed with custom or standardized atmospheric corrections (e.g., Sen2Cor) to ensure accurate radiometric signatures.

- **(Optional) Landsat 8/9 OLI:**
  - In some derived workflows, the WorldFloods pipeline and trained models can also ingest **Landsat 8/9** imagery (30 m resolution) for flood segmentation. This is enabled by a specialized version of the model that uses overlapping spectral bands between Sentinel-2 and Landsat.

### Original dataset

1. **Version 1.0.0**  
   - Released with the article: *Towards global flood mapping onboard low cost satellites with machine learning* (Mateo-García et al., 2021).  
   - Covers events from **2016 to 2019**, generated semi-automatically, with only validation and test subsets manually curated.  
   - Offers flood mask products with fewer curated maps than v2.  
   - Total download size: up to **~300 GB** if all subsets (train, val, test) are included.
   - **Google Drive**: A subset of data and pretrained models are shared in a **public Drive folder**.  
   - **Google Cloud Storage**: The GCS bucket is: `gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/`(Requires a GCP **requester pays** project).  


2. **Version 2.0.0**  
   - **Latest release** accompanying the article:  
     *Global flood extent segmentation in optical satellite images* (Portalés-Julià et al., 2023).  
   - Flood events from **2016 to 2023**, including manually curated flood masks based largely on Copernicus EMS products.  
   - Each sample has **two** binary reference channels: (1) clear/cloud and (2) land/water. This allows cloud-aware flood segmentation.  
   - Total download size: **~76 GB**.  
   - **Hugging Face**: The dataset is hosted under [isp-uv-es/WorldFloodsv2](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2). 
   - **Zenodo**: Manually curated flood masks, metadata, and additional products (including **Pakistan 2023 flood event** map) are available at [Zenodo, DOI: 10.5281/zenodo.8153514](https://doi.org/10.5281/zenodo.8153514).

## Taco dataset

To facilitate streamlined **data access and usage**, the WorldFloods dataset has been organized into [TACO](https://tacofoundation.github.io/) “tortillas.” Each tortilla file contains **three samples**:

1. **S2**: The multi-band Sentinel-2 image (Level-1C or Level-2A), harmonized to 10 m per pixel.  
2. **PERMANENTWATERJRC**: A reference mask derived from the JRC (Joint Research Centre) permanent water layer.  
3. **GT**: The curated flood extent mask, distinguishing water vs. land, as labeled by Copernicus EMS or other authoritative sources.

All three samples are co-registered over the same bounding box, enabling direct **pixel-wise** comparisons and model training. Each tortilla also includes STAC-like metadata describing the coordinate reference system (CRS), affine transform, time stamps, and other key descriptors.



## 🔄 Reproducible Example

<a target="_blank" href="https://colab.research.google.com/drive/1b_LfmYyJWId3aZ3ikdsB8-VTgAvUee59?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Load this dataset using the `tacoreader` library.

```python
import tacoreader
import rasterio as rio
import matplotlib.pyplot as plt

# Load the dataset
dataset = tacoreader.load("tacofoundation:worldfloods")

# Read a sample row
idx = 273
row = dataset.read(idx)
row_id = dataset.iloc[idx]["tortilla:id"]

# Retrieve the data
s2_path, per_path, gt_path  = row.read(0), row.read(1), row.read(2)
with rio.open(s2_path) as src_s2, rio.open(per_path) as src_per, rio.open(gt_path) as src_gt:
    window_s2 = rio.windows.Window(10300, 2900, 512, 512)
    window_per = rio.windows.Window(10300, 2900, 512, 512)
    window_gt = rio.windows.Window(10300, 2900, 512, 512)
    
    s2 = src_s2.read([2, 3, 4], window=window_s2)
    per = src_per.read(1, window=window_per)
    gt = src_gt.read(2, window=window_gt)

# Display
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(s2.transpose(1, 2, 0) / 2000)
axs[0].set_title('Sentinel-2')
axs[0].axis('off')
axs[1].imshow(gt, cmap='viridis') 
axs[1].set_title('Ground Truth')
axs[1].axis('off')
axs[2].imshow(per, cmap='coolwarm') 
axs[2].set_title('Permanent Water')
axs[2].axis('off')

plt.tight_layout()
plt.show()
```

<center>
    <img src='assets/worldfloods.png' alt='drawing' width='100%'/>
</center>
"""

bibtex_1 = """
@article{portales-julia_global_2023,
	title = {Global flood extent segmentation in optical satellite images},
	volume = {13},
	issn = {2045-2322},
	doi = {10.1038/s41598-023-47595-7},
	number = {1},
	urldate = {2023-11-30},
	journal = {Scientific Reports},
	author = {Portalés-Julià, Enrique and Mateo-García, Gonzalo and Purcell, Cormac and Gómez-Chova, Luis},
	month = nov,
	year = {2023},
	pages = {20316},
}
"""
bibtex_2 = """
@article{mateo-garcia_towards_2021,
	title = {Towards global flood mapping onboard low cost satellites with machine learning},
	volume = {11},
	issn = {2045-2322},
	doi = {10.1038/s41598-021-86650-z},
	number = {1},
	urldate = {2021-04-01},
	journal = {Scientific Reports},
	author = {Mateo-Garcia, Gonzalo and Veitch-Michaelis, Joshua and Smith, Lewis and Oprea, Silviu Vlad and Schumann, Guy and Gal, Yarin and Baydin, Atılım Güneş and Backes, Dietmar},
	month = mar,
	year = {2021},
	pages = {7249},
}
"""

# Create collection object with metadata
collection_object = tacotoolbox.datamodel.Collection(
    id="worldfloods",
    title="Global flood extent segmentation in optical satellite images",  # Update title accordingly
    dataset_version="1.0.0", # Update version accordingly
    description=description,  # Update description accordingly
    licenses=["cc-by-nc-4.0"], 
    extent={
        "spatial": [[-180.0, -90.0, 180.0, 90.0]],  # Define spatial extent
        "temporal": [["2016-01-01T00:00:00Z", "2023-12-31T23:59:59Z"]]  # Define temporal extent
    },
    providers=[
        {
            "name": "Universitat de València - Image & Signal Processing Group",
            "roles": ["producer"],
            "links": [
                {
                    "href": "https://isp.uv.es/",
                    "rel": "website",
                    "type": "text/html"
                }
            ],
        }
    ],
    keywords=["remote", "floods", "deep-learning", "sentinel-2"],
    task="semantic-segmentation",
    curators=[
        {
            "name": "Julio Contreras",
            "organization": "Image & Signal Processing",
            "email": ["julio.contreras@uv.es"],
            "links": [
                {
                    "href": "https://juliocontrerash.github.io/",
                    "rel": "homepage",
                    "type": "text/html"
                }
            ],
        }
    ],
    split_strategy="none", 
    discuss_link={
        "href": "https://huggingface.co/datasets/tacofoundation/worldfloods/discussions",
        "rel": "source",
        "type": "text/html"
    },
    raw_link={
        "href": "https://huggingface.co/datasets/tacofoundation/worldfloods",
        "rel": "source",
        "type": "text/html"
    },
    optical_data={"sensor": "sentinel2msi"}, # neon-ais
    labels={
        # Example label schema: land, flood-water, permanent-water, cloud, etc.
        "label_classes": [
            {"name": "no-flood-water", "category": 0, "description": "Non-water land pixels"},
            {"name": "flood-water",    "category": 1, "description": "Flooded areas"},
            {"name": "permanent-water","category": 2, "description": "Permanent water bodies"}
        ],
        "label_description": "Binary flood masks (land vs. water), plus permanent water references."
    },
    scientific = {
        "doi": "10.9999/zenodo.placeholder_worldfloods", 
        "citation": "WorldFloods dataset",  
        "summary": "WorldFloods dataset contains flood extent segmentation masks from Sentinel-2 imagery, used for flood mapping research. Public domain (CC0).",  # Dataset summary
        "publications": [
            {
                "doi": "10.1038/s41598-023-47595-7",
                "citation": bibtex_1,  
                "summary": "Portalés-Julià et al. (2023): Detailed flood extent segmentation using Sentinel-2 optical imagery.",  # Publication summary
            },
            {
                "doi": "10.1038/s41598-021-86650-z", 
                "citation": bibtex_2, 
                "summary": "Mateo-García et al. (2021): Using machine learning for global flood mapping onboard low-cost satellites.",  # Publication summary
            }
        ]
    }
)

# Create the final taco file with all samples and metadata
output_file = tacotoolbox.create(
    samples=samples_obj,
    collection=collection_object,
    output=pathlib.Path("/data/databases/Julio/Flood_kike/tacos") / "worldfloods.taco"
)




# Load and display the dataset for validation
dataset = tacoreader.load("tacofoundation:worldfloods")
idx = 0
row = dataset.read(idx)
s2_path, per_path, gt_path = row.read(0), row.read(1), row.read(2)

# Open Sentinel-2, Permanent Water, and Ground Truth files
with rio.open(s2_path) as src_s2, rio.open(per_path) as src_per, rio.open(gt_path) as src_gt:
    window_s2 = rio.windows.Window(512, 512, 512, 512)
    window_per = rio.windows.Window(512, 512, 512, 512)
    window_gt = rio.windows.Window(512, 512, 512, 512)
    
    s2 = src_s2.read([2, 3, 4], window=window_s2)
    per = src_per.read(1, window=window_per)
    gt = src_gt.read(2, window=window_gt)

# Plot the images
fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
axs[0].imshow(s2.transpose(1, 2, 0) / 2000)
axs[0].set_title('Sentinel-2')
axs[0].axis('off')
axs[1].imshow(gt, cmap='viridis') 
axs[1].set_title('Ground Truth')
axs[1].axis('off')
axs[2].imshow(per, cmap='coolwarm') 
axs[2].set_title('Permanent Water')
axs[2].axis('off')

plt.tight_layout()
plt.savefig("image.png")
plt.show()



