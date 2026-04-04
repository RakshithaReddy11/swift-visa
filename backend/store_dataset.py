import os
import hashlib
import sys
import chromadb
import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

USA_PATH = os.path.join(BASE_DIR, "data", "USA")
UK_PATH = os.path.join(BASE_DIR, "data", "uk_visa")
AUSTRALIA_PATH = os.path.join(BASE_DIR, "data", "AUSTRALIA")


print("Chroma path:", CHROMA_PATH)

client = chromadb.PersistentClient(path=CHROMA_PATH)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)


def process_country(country_name, data_path, collection_name):

    print("\n===============================")
    print(f"Processing {country_name} dataset")
    print("===============================\n")

    if not os.path.isdir(data_path):
        print(f"Error: dataset path not found for {country_name}: {data_path}")
        return

    collection = client.get_or_create_collection(collection_name)

    documents = []

    # Load PDFs
    for file in sorted(os.listdir(data_path)):

        if file.lower().endswith(".pdf"):

            path = os.path.join(data_path, file)

            print("Loading:", file)

            try:
                pdf = fitz.open(path)

                for page_number, page in enumerate(pdf, start=1):

                    try:
                        text = page.get_text("text") or ""
                    except Exception as page_error:
                        print(f"Skipping page {page_number} in {file}: {page_error}")
                        continue

                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file,
                                    "page": page_number,
                                    "country": country_name
                                }
                            )
                        )

                pdf.close()

            except Exception as file_error:
                print(f"Failed to parse {file}: {file_error}")

    print("\nTotal pages loaded:", len(documents))

    # Split into chunks
    chunks = splitter.split_documents(documents)

    print("Total chunks created:", len(chunks))

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Stable IDs
    ids = []

    for idx, chunk in enumerate(chunks):

        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "na")

        digest = hashlib.sha1(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:12]

        ids.append(f"{source}:{page}:{idx}:{digest}")

    print("\nStarting insertion...")

    batch_size = 1000
    total = len(texts)

    for i in range(0, total, batch_size):

        end = i + batch_size

        print(f"Inserting chunks {i} → {min(end, total)} of {total}")

        collection.upsert(
            documents=texts[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

    print(f"\n✅ {country_name} dataset stored successfully.\n")


def main() -> None:

    target = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "all"

    valid_targets = ["all", "us", "usa", "uk", "australia"]

    if target in ("all", "us", "usa"):
        process_country(
            "USA",
            USA_PATH,
            "us_visa_collection"
        )

    if target in ("all", "uk"):
        process_country(
            "UK",
            UK_PATH,
            "uk_visa_collection"
        )

    if target in ("all", "australia"):
        process_country(
            "AUSTRALIA",
            AUSTRALIA_PATH,
            "australia_collection"
        )

    if target not in valid_targets:
        print("Unknown option:", target)
        print("Usage: python store_dataset.py [all|us|uk|australia]")
        raise SystemExit(1)

    print("\nAll selected datasets stored successfully!")


if __name__ == "__main__":
    main()