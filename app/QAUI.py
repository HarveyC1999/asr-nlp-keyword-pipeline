"""
Created on Thu Feb  6 14:37:03 2025
@author: HarveyC
"""

import tkinter as tk
from tkinter import ttk
import cx_Oracle
import pandas as pd
from datetime import datetime
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
import os 


## DB login
def get_key(key_file):
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key


def decrypt_data(encrypt_msg):
    private_key = get_key(r'E:\QAs\rsa_private_key.pem')
    cipher = PKCS1_cipher.new(private_key)
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg), 0)
    return back_text.decode('utf-8')


encrypt_msg = 'RSA encrypted string'

""" Retrieve data from Oracle DB and filter based on conditions """
login = decrypt_data(encrypt_msg)
conn = cx_Oracle.connect(login, encoding='UTF-8', nencoding='UTF-8')


# Database query function
def get_data(start_date='', end_date='', agt_id='', product_type=''):
    """ Retrieve data from Oracle DB and apply filters """
    login = decrypt_data(encrypt_msg)
    conn = cx_Oracle.connect(login, encoding='UTF-8', nencoding='UTF-8')

    query = """select to_char(f.create_date,'YYYYMMDD') SEARCH_DATE, m.Agent_Id, m.prod_type, m.product_code,
               f.FILE_NAME, 'D:/QA/finished'||f.FILE_PATH
               from ims.stt_chk_lst_m m
               left join ims.stt_chk_lst_file f
               on m.stt_lst_seq = f.stt_lst_seq
               where stt_flag = 'STT' """

    if start_date:
        query += f" AND to_char(f.create_date,'YYYYMMDD') >= {start_date}"
    if end_date:
        query += f" AND to_char(f.create_date,'YYYYMMDD') <= {end_date}"    
    if agt_id:
        query += f" AND m.Agent_Id = '{agt_id}'"
    if product_type and product_type != "All":
        query += f" AND m.prod_type = '{product_type}'"

    query += " and file_name is not null ORDER BY m.STT_LST_SEQ, CAST(SEQ AS DECIMAL) ASC"

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def update_table():
    """ Refresh table display """
    for row in tree.get_children():
        tree.delete(row)

    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    agt_id = agt_id_entry.get()
    product_type = product_type_var.get()

    data = get_data(start_date, end_date, agt_id, product_type)

    for _, row in data.iterrows():
        tree.insert('', 'end', values=row.tolist())


def on_treeview_click(event):
    """ Handle Treeview click event """
    item = tree.identify('item', event.x, event.y)
    col = tree.identify_column(event.x)

    if col == '#6':  # File path column
        file_path = tree.item(item, 'values')[5]

        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            tk.messagebox.showerror("Error", f"File path does not exist: {file_path}")


# Create GUI
root = tk.Tk()
root.title("Oracle Data Filter Viewer")

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)


# Filter section
tk.Label(root, text="Start Date (YYYYMMDD):").grid(row=0, column=0)
start_date_entry = tk.Entry(root)
start_date_entry.grid(row=0, column=1)

tk.Label(root, text="End Date (YYYYMMDD):").grid(row=0, column=2)
end_date_entry = tk.Entry(root)
end_date_entry.grid(row=0, column=3)

tk.Label(root, text="AGENT_ID:").grid(row=0, column=4)
agt_id_entry = tk.Entry(root)
agt_id_entry.grid(row=0, column=5)

tk.Label(root, text="LF/HP:").grid(row=0, column=6)
product_type_var = tk.StringVar()
product_type_dropdown = ttk.Combobox(
    root,
    textvariable=product_type_var,
    values=["All", "LF", "HP"]
)
product_type_dropdown.grid(row=0, column=7)


# Update button
update_btn = tk.Button(root, text="Update", command=update_table)
update_btn.grid(row=0, column=8)


# Table + scrollbar
tree_frame = tk.Frame(root)
tree_frame.grid(row=1, column=0, columnspan=9, sticky='nsew')

tree_scroll_y = tk.Scrollbar(tree_frame, orient="vertical")
tree_scroll_y.pack(side="right", fill="y")

tree = ttk.Treeview(
    tree_frame,
    columns=("Date", "AGENT_ID", "Insurance Type", "Product Name", "File Name", "File Path"),
    show="headings",
    yscrollcommand=tree_scroll_y.set
)

tree.heading("Date", text="Date")
tree.heading("AGENT_ID", text="AGENT_ID")
tree.heading("Insurance Type", text="Insurance Type")
tree.heading("Product Name", text="Product Name")
tree.heading("File Name", text="File Name")
tree.heading("File Path", text="File Path")

tree.pack(side="left", fill="both", expand=True)

for col in tree["columns"]:
    tree.column(col, width=100, minwidth=100, stretch=True)

tree_scroll_y.config(command=tree.yview)

tree.bind("<ButtonRelease-1>", on_treeview_click)

update_table()


# Run application
root.mainloop()