# Data

This project uses publicly available e-commerce datasets for customer segmentation and trajectory modeling.

## Final Dataset

The final experiments were performed using the **Online Retail II** dataset.

The dataset contains transactional records including:

- Customer ID
- Invoice number
- Invoice date
- Quantity
- Product price
- Product description
- Country

The raw dataset is **not included in this repository**.

## Dataset Usage

The data was processed to:

- remove invalid or cancelled transactions
- remove records without customer IDs
- calculate transaction value
- aggregate purchases by month
- generate RFM-based features
- identify customers with activity across multiple months
- create customer behavior sequences for the Sequential VAE

The final trajectory-modeling cohort contained customers active in at least **3 different months**.

## Olist Dataset

The **Olist Brazilian E-Commerce Dataset** was also evaluated during the dataset viability stage.

It was not used for the final trajectory-modeling experiments because too few customers had purchase activity across enough separate months.

## How to Use

1. Download the Online Retail II dataset.
2. Place the raw dataset in this folder.
3. Update the dataset path in `01_data_and_viability.ipynb` if necessary.
4. Run the notebooks in numerical order.


