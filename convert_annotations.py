import json
import re

input_file = "data/raw_annotations.jsonl"
output_file = "data/annotations_converted.jsonl"


def parse_metadata(raw_text):
    # Extract invoice_no example: looks for patterns like FA01/2015/056091
    invoice_no_match = re.search(r"(FA\d{2}/\d{4}/\d+)", raw_text)
    invoice_no = invoice_no_match.group(1) if invoice_no_match else ""

    # Extract date: pattern dd/mm/yyyy or yyyy-mm-dd or dd-mm-yyyy
    date_match = re.search(r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", raw_text)
    date = date_match.group(1) if date_match else ""

    # Extract company name - just an example: company name is the first non-empty line
    lines = raw_text.strip().splitlines()
    company = lines[0].strip() if lines else ""

    # GSTIN: pattern GSTIN or 15 alphanumeric characters
    gstin_match = re.search(r"\b([0-9A-Z]{15})\b", raw_text)
    gstin = gstin_match.group(1) if gstin_match else ""

    # Extract billing and shipping address - dummy: take lines after "Adresse de facturation" and "Adresse de livraison"
    billing_address = ""
    shipping_address = ""

    billing_match = re.search(r"Adresse de facturation\s*(.*?)Adresse de livraison", raw_text, re.DOTALL)
    if billing_match:
        billing_address = billing_match.group(1).strip().replace('\n', ', ')

    shipping_match = re.search(r"Adresse de livraison\s*(.*?)ID client", raw_text, re.DOTALL)
    if shipping_match:
        shipping_address = shipping_match.group(1).strip().replace('\n', ', ')

    return {
        "invoice_no": invoice_no,
        "date": date,
        "company": company,
        "gstin": gstin,
        "billing_address": billing_address,
        "shipping_address": shipping_address
    }


def parse_items(raw_text):
    # Extract items from table lines by looking for lines with columns separated by spaces/tabs
    items = []
    lines = raw_text.splitlines()
    item_section = False

    for line in lines:
        # Detect start of items table by keywords, e.g. Description and Quantity in header
        if re.search(r"Description.*Quantité", line, re.IGNORECASE):
            item_section = True
            continue
        if item_section:
            if not line.strip():
                # Blank line means end of items table
                break
            # Split by multiple spaces or tabs - naive approach
            parts = re.split(r"\s{2,}|\t", line.strip())
            if len(parts) >= 7:
                # Map columns according to your table example
                name = parts[1]
                try:
                    quantity = float(parts[2].replace(',', ''))
                except:
                    quantity = 0.0
                try:
                    unit_price = float(parts[3].replace('€', '').replace(',', '').strip())
                except:
                    unit_price = 0.0
                try:
                    price_without_gst = float(parts[4].replace('€', '').replace(',', '').strip())
                except:
                    price_without_gst = 0.0
                try:
                    gst_amount = float(parts[6].replace('€', '').replace(',', '').strip())
                except:
                    gst_amount = 0.0
                try:
                    price = float(parts[7].replace('€', '').replace(',', '').strip())
                except:
                    price = 0.0
                # Discount and category missing in example, set dummy 0 and default
                discount_amount = 0.0
                category = "General"

                item = {
                    "name": name,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "price_without_gst": price_without_gst,
                    "gst_amount": gst_amount,
                    "price": price,
                    "discount_amount": discount_amount,
                    "category": category
                }
                items.append(item)
    return items


def parse_summary(items):
    subtotal = sum(item["price_without_gst"] for item in items)
    total_gst = sum(item["gst_amount"] for item in items)
    total_discount = sum(item["discount_amount"] for item in items)
    total_amount = sum(item["price"] for item in items)

    return {
        "subtotal": subtotal,
        "total_gst": total_gst,
        "total_discount": total_discount,
        "total_amount": total_amount
    }


def parse_categories(items):
    categories = {}
    for item in items:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "total": 0.0, "gst": 0.0}
        categories[cat]["count"] += item["quantity"]
        categories[cat]["total"] += item["price_without_gst"]
        categories[cat]["gst"] += item["gst_amount"]
    return categories


def convert_line(line):
    data = json.loads(line)
    raw_text = data.get("text", "")

    metadata = parse_metadata(raw_text)
    items = parse_items(raw_text)
    summary = parse_summary(items)
    categories = parse_categories(items)

    converted = {
        "extracted_data": {
            "metadata": metadata,
            "items": items,
            "summary": summary,
            "categories": categories,
            "raw_text": raw_text
        }
    }
    return json.dumps(converted, ensure_ascii=False)


def main():
    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            converted_line = convert_line(line)
            outfile.write(converted_line + "\n")
    print(f"Conversion completed. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
