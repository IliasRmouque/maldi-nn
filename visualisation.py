#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# data = os.listdir("../data")
# #%%
# for file in data:
#     if file.endswith(".npz"):
#         specs = np.load(os.path.join("../data", file))
#         mzs = specs["mzs"]
#         intensities = specs["spectrums"]
#         print(f"Loaded {file} with {len(mzs)} spectra.")
        
#         # Plotting the first spectrum as an example
        
#         plt.scatter(mzs[:10], intensities[:10], alpha=0.5, label=file)

        
# plt.xlabel("m/z")
# plt.ylabel("Intensity")
# plt.show()
# # %%
# annotations_files = []    
# for root,dirs, files in  os.walk("/media/ilias/Crucial X9/Recal/"):
#     for file in files:
#         if file.endswith(".parquet"):
#             annotations_files.append(os.path.join(root, file))


# %%
# for file_parquet in annotations_files:
#     name_image = file_parquet.split("/")[-1].split(".")[0]
#     for file in data:
#         if file.endswith(".npz"):
#             if name_image.lower() in file.lower():
#                 print(f"Found {name_image} in {file}")
#                 spectrum_files = np.load(os.path.join("../data", file))
#                 mzs = spectrum_files["mzs"]
#                 intensities = spectrum_files["spectrums"]
#                 xs = spectrum_files["x"]
#                 ys = spectrum_files["y"]
#                 print(f"Loaded {file} with {len(mzs)} spectra.")
                
#                 df = pd.read_parquet(file_parquet)

#                 # put the annotations term in the npz file
#                 annotations = df["annotation_term"].values
#                 xs_annotation = df["x"].values
#                 ys_annotation = df["y"].values

#                 #put an index on xs/ys = annotations
#                 index = pd.MultiIndex.from_arrays([xs_annotation, ys_annotation], names=["x", "y"])
#                 annotations_df = pd.DataFrame(annotations, index=index)


#                 point_in= []
#                 point_out = []
#                 for i in range(len(xs)):
#                     if (xs[i], ys[i]) in annotations_df.index:
#                         point_in.append((xs[i], ys[i])) 
#                     else:
#                         point_out.append((xs[i], ys[i]))

#                 #keep only the points that are in the annotations
                
#                 mzs = [mzs[i] for i in range(len(mzs)) if (xs[i], ys[i]) in annotations_df.index]
#                 intensities = [intensities[i] for i in range(len(intensities)) if (xs[i], ys[i]) in annotations_df.index]
#                 xs = [point[0] for point in point_in]
#                 ys = [point[1] for point in point_in]
#                 annotations_df = annotations_df.loc[point_in]

#                 #save the annotations in the npz file
#                 np.savez_compressed(os.path.join("../data", file ), mzs=np.array(mzs), spectrums=np.array(intensities), x=np.array(xs), y=np.array(ys), annotations=annotations_df)

# # %%
# data = os.listdir("../data")

# #load the npz file and display by annotations
# for file in data:
#     if file.endswith(".npz"):
#         specs = np.load(os.path.join("../data", file), allow_pickle=True)
#         mzs = specs["mzs"]
#         intensities = specs["spectrums"]
#         xs = specs["x"]
#         ys = specs["y"]
#         annotations= specs["annotations"]
#         print(f"Loaded {file} with {len(mzs)} spectra.")
        
#         real_annotations = []
#         for i, a in enumerate(annotations):
#             if "cck" in a[0].lower():
#                 real_annotations.append("cck")
#             elif "chc" in a[0].lower():
#                 real_annotations.append("chc")
#             elif "fnt" in a[0].lower():
#                 real_annotations.append("fnt")
#             else:
#                 real_annotations.append("other") 

#         # Plotting the first spectrum as an example
#         for a in np.unique(real_annotations):
#             i = np.where(annotations == a)[0][:10]
#             plt.scatter(mzs[i], intensities[i], alpha=0.5, label=a)
#         plt.legend()
#         plt.xlabel("m/z")
#         plt.ylabel("Intensity")
#         plt.title(file)
# plt.show()
# # %%
# #put all the npz files together in one file
# # Initialize lists to store data from all files
# all_mzs = []
# all_intensities = []
# all_xs = []
# all_ys = []
# all_annotations = []
# origins = []

# # Process each npz file
# for file in data:
#     if file.endswith(".npz"):
#         specs = np.load(os.path.join("../data", file), allow_pickle=True)
#         mzs = specs["mzs"]
#         intensities = specs["spectrums"]
#         xs = specs["x"]
#         ys = specs["y"]
#         # Create array of filenames with same length as ys
#         origin = [file] * len(ys)
#         annotations = specs["annotations"]
        
#         # Add data from this file
#         all_mzs.append(mzs)
#         all_intensities.append(intensities)
#         all_xs.append(xs)
#         all_ys.append(ys)
#         all_annotations.append(annotations)
#         origins += origin
        
#         print(f"Added data from {file}")

# # Combine arrays
# all_mzs = np.concatenate(all_mzs)
# all_intensities = np.concatenate(all_intensities)
# all_xs = np.concatenate(all_xs)
# all_ys = np.concatenate(all_ys)

# # For annotations, try simple concatenation first
# try:
#     all_annotations = np.concatenate(all_annotations)
# except Exception as e:
#     print(f"Could not concatenate annotations: {e}")
#     # Convert to object array if concatenation fails
#     all_annotations = np.array(sum([list(a) for a in all_annotations], []), dtype=object)

# # Save combined data
# np.savez_compressed("../data/all_spectra.npz", 
#                    mzs=all_mzs, 
#                    spectrums=all_intensities, 
#                    x=all_xs, 
#                    y=all_ys, 
#                    annotations=all_annotations,
#                    origins=origins)

# print(f"Saved combined data with {len(all_mzs)} total entries to all_spectra.npz")
                

# # %%

# # # Load the combined data
# combined_data = np.load("../data/all_spectra.npz", allow_pickle=True)

# #filter the annotations only the ones that in lower contains cck, chc or fnt should be kept
# annotations = combined_data["annotations"]
# #indices = [i for i, a in enumerate(annotations) if "cck" in a[0].lower() or "chc" in a[0].lower() or "fnt" in a[0].lower()]
# #replace the annotations with cck, chc or fnt
# real_annotations = []
# for i, a in enumerate(annotations):
#     if "cck" in a[0].lower():
#         real_annotations.append("cck")
#     elif "chc" in a[0].lower():
#         real_annotations.append("chc")
#     elif "fnt" in a[0].lower():
#         real_annotations.append("fnt")
#     else:
#         real_annotations.append("other")


# #remove other
# indices = [i for i, a in enumerate(real_annotations) if a == "other"]
# #remove the indices from the combined data
# all_mzs = np.delete(combined_data["mzs"], indices, axis=0)
# all_intensities = np.delete(combined_data["spectrums"], indices, axis=0)
# all_xs = np.delete(combined_data["x"], indices, axis=0)
# all_ys = np.delete(combined_data["y"], indices, axis=0)
# all_annotations = np.delete(real_annotations, indices, axis=0)
# origins = np.delete(combined_data["origins"], indices, axis=0)



# # %%
# # Load the combined data
combined_data = np.load("../data/all_spectra.npz", allow_pickle=True)
annotations = combined_data["annotations"]
for i, a in enumerate(annotations):
    if "cck" in a[0].lower():
        annotations[i] = "cck"
    elif "chc" in a[0].lower():
        annotations[i] = "chc"
    elif "fnt" in a[0].lower():
        annotations[i] = "fnt"
    else:
        annotations[i] = "other"

for annotation in np.unique(annotations):
    if annotation == "other":
        continue
    indices = np.random.choice(np.where(annotations == annotation)[0], size=20, replace=False)
    plt.scatter(combined_data["mzs"][indices], combined_data["spectrums"][indices], alpha=0.5, label=annotation)

plt.xlabel("m/z")
plt.ylabel("Intensity")
plt.title("All Spectra")
plt.legend()
plt.show()

