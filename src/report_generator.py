import pandas as pd

# data load
df = pd.read_csv('sales_data.csv')
print("Data Loaded Successfully")

# calculate total sales
total_sales = df['sales'].sum()
print(f"Total Sales: {total_sales}")

# calculate average sales
average_sales = df['sales'].mean()
print(f"Average Sales: {average_sales}")

# sales by product
sales_by_product = df.groupby('product')['sales'].sum().reset_index()
print("Sales by Product:")

# lets create html report
html_report = f"""
<html>
<head><title>Sales Report</title></head>
<body>
<h1>Sales Report</h1>
<p>Total Sales: {total_sales}</p>
<p>Average Sales: {average_sales}</p>
<h2>Sales by Product</h2>
<table border="1">
<tr><th>Product</th><th>Sales</th></tr>
"""
for index, row in sales_by_product.iterrows():
    html_report += f"<tr><td>{row['product']}</td><td>{row['sales']}</td></tr>"
html_report += """
</table>
</body>
</html>
"""
with open('sales_report.html', 'w') as f:
    f.write(html_report)
print("Sales report generated: sales_report.html")