import mysql.connector
import pandas as pd
from decimal import Decimal
class Data_getter():
    def get_data_from_server(b_id):
        # Define the connection parameters
        host = "159.138.104.192"
        user = "storemate_ml"
        password = "bTgZd77VpD^o4Ai6Dw9xs9"
        database = "lite_version"  # Replace with the name of the database you want to connect to

        # Create a connection to the MySQL server
        try:
            # Create a connection to the MySQL server
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )

            if connection.is_connected():
                print("Connected to MySQL database")

                # Create a cursor object for executing SQL queries
                cursor = connection.cursor()

                # Define the SQL SELECT query
                sql_query = """
        SELECT
          `b`.`id` AS `business_id`,
          `b`.`name` AS `business_name`,
          `p`.`name` AS `product_name`,
          `p`.`type` AS `product_type`,
          `c1`.`name` AS `category_name`,
          `pv`.`name` AS `product_variation`,
          `v`.`name` AS `variation_name`,
          `v`.`sub_sku`,
          `c`.`name` AS `customer`,
          `c`.`contact_id`,
          `t`.`id` AS `transaction_id`,
          `t`.`invoice_no`,
          `t`.`transaction_date` AS `transaction_date`,
          (transaction_sell_lines.quantity - transaction_sell_lines.quantity_returned) AS sell_qty,
          `u`.`short_name` AS `unit`,
          transaction_sell_lines.unit_price_inc_tax,
        transaction_sell_lines.unit_price_before_discount
        FROM `transaction_sell_lines`
          INNER JOIN `transactions` AS `t`
            ON `transaction_sell_lines`.`transaction_id` = `t`.`id`
          INNER JOIN `variations` AS `v`
            ON `transaction_sell_lines`.`variation_id` = `v`.`id`
          LEFT JOIN `transaction_sell_lines_purchase_lines` AS `tspl`
            ON `transaction_sell_lines`.`id` = `tspl`.`sell_line_id`
          LEFT JOIN `purchase_lines` AS `pl`
            ON `tspl`.`purchase_line_id` = `pl`.`id`
          INNER JOIN `product_variations` AS `pv`
            ON `v`.`product_variation_id` = `pv`.`id`
          INNER JOIN `contacts` AS `c`
            ON `t`.`contact_id` = `c`.`id`
          INNER JOIN `products` AS `p`
            ON `pv`.`product_id` = `p`.`id`
          LEFT JOIN `business` AS `b`
            ON `p`.`business_id` = `b`.`id`
          LEFT JOIN `categories` AS `c1`
            ON `p`.`category_id` = `c1`.`id`
          LEFT JOIN `tax_rates`
            ON `transaction_sell_lines`.`tax_id` = `tax_rates`.`id`
          LEFT JOIN `units` AS `u`
            ON `p`.`unit_id` = `u`.`id`
          LEFT JOIN `transaction_payments` AS `tp`
            ON `tp`.`transaction_id` = `t`.`id`
          LEFT JOIN `transaction_sell_lines` AS `tsl`
            ON `transaction_sell_lines`.`parent_sell_line_id` = `tsl`.`id`
        WHERE `t`.`type` = 'sell'
        AND `t`.`status` = 'final'
        AND t.business_id = """+str(b_id)+"""
        GROUP BY `b`.`id`,
                 `transaction_sell_lines`.`id`;
        #find bussiness
        SELECT b.id AS 'Bussiness ID', b.name AS 'Bussiness Name','>>'
        ,bl.id AS 'Location ID', bl.name AS 'Location Name' FROM business b 
        INNER JOIN business_locations bl ON b.id =bl.business_id WHERE b.name LIKE '%sup%' 
                """

                # Execute the SQL query
                cursor.execute(sql_query)

                results = cursor.fetchall()
                results = [tuple(
                    float(val) if isinstance(val, Decimal) else val for val in row
                ) for row in results]
                # Display the results
                #for row in results:
                    #print(row)  # You can process the results as needed

                # Close the cursor and connection

                # Create a DataFrame
                columns = ["business_id","business_name","product_name","product_type","category_name","product_variation","variation_name","sub_sku","customer","contact_id","transaction_id","invoice_no","transaction_date","sell_qty","unit","cost_price","selling_price"]
                df = pd.DataFrame(results, columns=columns)
                df = df[:2000]
                #print(df.to_string())
                df.to_csv("salesdata/"+str(b_id)+".csv")
                # Display the DataFrame as a table
                #dict_list = df.to_dict(orient='records')
                
                cursor.close()
                connection.close()
                return "salesdata/"+str(b_id)+".csv"

        except mysql.connector.Error as e:
            return ("Error:", e)
