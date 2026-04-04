import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collections = client.list_collections()

print("\nAvailable Collections:\n")

for col in collections:
    collection = client.get_collection(col.name)

    print("Collection Name:", col.name)
    print("Total Chunks:", collection.count())

    # Fetch only a small sample to avoid SQL variable limits

    try:
        data = collection.get(limit=100, include=["metadatas"])
        # Collect unique file names from the 'source' field in metadata
        sources = set()
        for meta in data["metadatas"]:
            source = meta.get("source")
            if source:
                sources.add(source)
        print("Original Files in Collection (sampled):")
        for source in sorted(sources):
            print("-", source)
    except Exception as e:
        print("Error fetching file names:", e)

    print("\n---------------------\n")