# Python libraries
import json
import tabulate

# Defining specifications for container images and container registries
ONLY_IMAGES_OF_INTEREST = 1
IMAGES_OF_INTEREST = [
    ###########################
    #### Operating Systems ####
    ###########################
    "debian",
    "centos",
    "ubuntu",
    "fedora",
    ###########################
    #### Language Runtimes ####
    ###########################
    "golang",
    "rust",
    "python",
    "ruby",
    ##############################
    #### Generic Applications ####
    ##############################
    "tomcat",
    "jetty",
]

with open("top150_images_dockerhub.json", "r", encoding="UTF-8") as read_file:
    data = json.load(read_file)

container_images = []
for image_metadata in data:
    if "layers" in image_metadata:
        if ONLY_IMAGES_OF_INTEREST and image_metadata["name"] not in IMAGES_OF_INTEREST:
            continue

        image = {
            "name": image_metadata["name"],
            "digest": image_metadata["digest"],
            "description": image_metadata["description"],
            "architecture": image_metadata["architecture"],
            "star_count": image_metadata["star_count"],
            "pull_count": image_metadata["pull_count"],
            "size": image_metadata["size"],
            "layers": [],
        }
        for layer in image_metadata["layers"]:
            if layer["size"] >= 1:
                image["layers"].append(layer)

        if len(image["layers"]) > 0:
            container_images.append(image)


for image in container_images:
    print(f"{image['name']}. Size: {image['size']}. Layers: {len(image['layers'])}")


# Collecting layer information
container_layers = {}
for container_image in container_images:
    for layer in container_image["layers"]:
        if layer["digest"] not in container_layers:
            container_layers[layer["digest"]] = {
                "digest": layer["digest"],
                "size": layer["size"],
                "occurrences": 1,
            }
        else:
            container_layers[layer["digest"]]["occurrences"] += 1

# Grouping unique and shared layers
unique_layers = []
shared_layers = []
for layer_digest, layer_metadata in container_layers.items():
    if layer_metadata["occurrences"] == 1:
        unique_layers.append(
            {
                "digest": layer_metadata["digest"],
                "size": layer_metadata["size"],
            }
        )
    else:
        shared_layers.append(
            {
                "digest": layer_metadata["digest"],
                "size": layer_metadata["size"],
            }
        )

# Calculating the amount of unique and shared layers in each container image
for container_image in container_images:
    container_image["size_shared_layers"] = 0
    container_image["size_unique_layers"] = 0
    for layer in container_image["layers"]:
        if any(layer["digest"] == l["digest"] for l in shared_layers):
            container_image["size_shared_layers"] += layer["size"]
        else:
            container_image["size_unique_layers"] += layer["size"]

# Aggregating metrics about the container layers
size_unique_layers = sum([layer["size"] for layer in unique_layers])
size_shared_layers = sum([layer["size"] for layer in shared_layers])
total_size_layers = size_unique_layers + size_shared_layers

print("==========================")
print("==== CONTAINER LAYERS ====")
print("==========================")
print(f"Total number of layers: {len(unique_layers) + len(shared_layers)}")
print(f"Number of unique layers: {len(unique_layers)}")
print(f"Number of shared layers: {len(shared_layers)}")
print(f"Total Size of layers: {size_unique_layers + size_shared_layers}")
print(f"Size of unique layers: {size_unique_layers} ({round(size_unique_layers / total_size_layers * 100, 2)}% of total)")
print(f"Size of shared layers: {size_shared_layers} ({round(size_shared_layers / total_size_layers * 100, 2)}% of total)")

print("\n")

print("==========================")
print("==== CONTAINER IMAGES ====")
print("==========================")
container_images_summary = []
for index, container_image in enumerate(container_images, 1):
    image_metadata = {
        "name": container_image["name"],
        "number_of_layers": len(container_image["layers"]),
        "total_size": container_image["size"],
        "avg_layer_size": container_image["size"] / len(container_image["layers"]),
        "size_unique": container_image["size_unique_layers"],
        "size_shared": container_image["size_shared_layers"],
    }
    container_images_summary.append(image_metadata)

container_images_summary = sorted(container_images_summary, key=lambda image: image["size_shared"])

header = container_images_summary[0].keys()
rows = [x.values() for x in container_images_summary]
print(tabulate.tabulate(rows, header))

# Exporting the metadata about the container images
container_images_dataset = []
for container_image in container_images:
    container_layers = [
        {
            "digest": layer["digest"],
            "size": layer["size"],
        }
        for layer in container_image["layers"]
    ]
    container_image_metadata = {
        "name": container_image["name"],
        "digest": container_image["digest"],
        "layers": container_layers,
    }
    container_images_dataset.append(container_image_metadata)


with open("container_images.json", "w") as outfile:
    json.dump(container_images_dataset, outfile)
