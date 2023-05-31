import pandas as pd
import numpy as np
import pickle
import pyarrow.parquet as pq
import streamlit as st
import plotly.graph_objs as go
import matplotlib as plt
import seaborn as sns
import plotly_express as px
from PIL import Image
import requests
from streamlit_option_menu import option_menu
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import base64
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
from statsmodels.tsa.arima.model import ARIMA
import re
import requests
import plotly.graph_objects as go
import plotly.colors


# Visualization
import matplotlib.pyplot as plt

from tools.my_functions import (
    modify_data_types,
    transformar_columnas_datetime,
    get_column_types,
    top_df_final,
    barv_plotly,
    pie_graph,
    sec_bar_mdf,
    line_graph,
    top_df_simple,
    line_graph_mult,
)

import warnings

# ...

# Antes de llamar a la función que produce la advertencia
warnings.filterwarnings(
    "ignore", message="Parsing dates in DD/MM/YYYY format when dayfirst=False"
)

# Llamar a la función que produce la advertencia
# ...

# Después de llamar a la función que produce la advertencia
warnings.filterwarnings(
    "default", message="Parsing dates in DD/MM/YYYY format when dayfirst=False"
)

pd.options.display.max_columns = None
st.set_page_config(
    page_title="EDA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


class EDA:
    def __init__(self):
        self.df = None
        self.separator = ","
        self.bg_image = None

    def add_bg_color(self):
        title_html = f"""
            <div style="text-align: center;">
                <h1 style="font-family: Helvetica Neue, serif;
               font-style: italic;
               font-weight: bold;
               font-size: 35px;
               color: white;
               background-repeat: no-repeat;
               background-size: cover;
               padding: 20px;">Business Intelligence EDA</h1>
            </div>

        """

        st.markdown(title_html, unsafe_allow_html=True)

        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #1a1a1a; /* Azul oscuro mezclado con gris */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def load_file(_self):
        uploaded_file = st.file_uploader(
            "Cargar archivo", type=["xlsx", "csv", "pickle"]
        )
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == "xlsx":
                _self.dataset = pd.read_excel(uploaded_file)
            elif file_extension == "csv":
                if uploaded_file.size == 0:
                    _self.dataset = None
                else:
                    _self.dataset = pd.read_csv(uploaded_file, sep=_self.separator)
            elif file_extension == "pickle":
                _self.dataset = pickle.load(uploaded_file)
        else:
            _self.dataset = None

    def visualize_data(self):
        if self.dataset is not None:
            modify_data_types(self.dataset, categories_number=150)
            transformar_columnas_datetime(self.dataset)

            # options = {
            #     "EDA": "EDA 🔍",
            #     "Power-Bi": "Power-Bi 📈",
            #     "Insights": "Insights ✅",
            # }

            # # Agrega la lista desplegable con los emojis
            # selected_option = st.selectbox(
            #     "Selecciona una opción",
            #     list(options.keys()),
            #     format_func=lambda x: options[x],
            # )
            # if selected_option == "Contexto":
            #     # Establecer el estilo de letra más grande
            #     st.markdown(
            #         "<style>h1 {font-size: 40px !important;}</style>",
            #         unsafe_allow_html=True,
            #     )

            #     # Cambiar el estilo de letra a una más elegante
            #     st.markdown(
            #         "<style>h1 {font-family: 'Garamond', serif;}</style>",
            #         unsafe_allow_html=True,
            #     )

            #     # Cambiar el estilo de letra a una más elegante para el texto extraído
            #     st.markdown(
            #         "<style>.extracted-text {font-family: 'Garamond', serif;}</style>",
            #         unsafe_allow_html=True,
            #     )

            #     # Crear el título con el estilo de letra y formato deseado
            #     titulo = "<h1 style='text-align: center; color: #88ff88; font-style: italic;'>Giant Supermarket</h1>"
            #     st.markdown(titulo, unsafe_allow_html=True)

            #     # Resto de tu código...
            #     url = (
            #         self.url
            #     )  # Asumiendo que la URL está almacenada en el atributo "url" de la clase
            #     response = requests.get(url)
            #     soup = BeautifulSoup(response.content, "html.parser")
            #     elements = soup.select(
            #         ".title-after_title"
            #     )  # Aplicar el selector .title-after_title

            #     extracted_text = [element.get_text(strip=True) for element in elements]

            #     # Aplicar el estilo de letra y formato deseado al texto extraído
            #     styled_text = (
            #         "<p class='extracted-text' style='font-size: 20px;'>"
            #         + "</p><p class='extracted-text' style='font-size: 20px;'>".join(
            #             extracted_text
            #         )
            #         + "</p>"
            #     )
            #     st.markdown(styled_text, unsafe_allow_html=True)

            #     st.dataframe(self.dataset)

            # if selected_option == "EDA":
            self.df = self.dataset.copy()
            # modify_data_types(self.df, categories_number=150)
            # st.write(self.df.dtypes)

            (
                self.date_cols,
                self.num_cols,
                self.obj_cols,
                self.cat_cols,
            ) = get_column_types(self.df)

            # self.obj_cols = self.df.select_dtypes(include=["object"]).columns
            # self.cat_cols = self.df.select_dtypes(include=["category"]).columns
            # self.num_cols = self.df.select_dtypes(include=["int", "float"]).columns
            # self.date_cols = self.df.select_dtypes(
            #     include=["datetime64[ns]"]
            # ).columns
            # # self.date_cols = self.date_cols.append(self.num_cols[-7:-1])
            # # self.num_cols = self.num_cols[0:-7]
            self.str_list = list(self.obj_cols) + list(self.cat_cols)
            self.cat_list = list(self.cat_cols) + list(self.obj_cols)
            self.asce_list = [False, True]

            with st.sidebar:
                self.main_column = st.selectbox("Categorical column #1", self.str_list)

                self.main_cat_col = st.selectbox("Categorical column #2", self.cat_cols)
                self.main_num_col = st.selectbox("Numeric column", self.num_cols)
                self.main_date_col = st.selectbox("Datetime column", self.date_cols)

                # formato_ship_date = determinar_formato_fecha(
                #     self.df, self.main_date_col
                # )

                # transformar_columnas_datetime(self.df, self.main_date_col)

                # st.write(formato_ship_date)

                # self.main_date_col = pd.to_datetime(self.main_date_col)

                self.top = 10
                self.orden = 1
                # e el orden del dataframe", list(range(1, 7)))
                self.ascen = st.selectbox("Top/Last", [False, True])
                self.ascend = "Top"
                if self.ascen == True:
                    self.ascend = "Last"

                self.size_title = 2
                self.color = st.selectbox(
                    " Categorical graphic color",
                    ["Emrld", "Turbo", "Inferno", "Plasma", "Magma", "Viridis"],
                    index=0,
                )
                self.height = 300
                self.width = 10000

                # self.ori = st.selectbox(
                #     "Orientación del gráfico",
                #     ["Vertical", "Horizontal"],
                #     index=0,
                # )

                # if self.ori == "Vertical":
                #     self.ori = "v"
                # else:
                #     self.ori = "h"

                # Opciones de gráficas
                # plot_options = {
                #     "Gráfico de barras": barv_plotly,
                #     "Gráfico tipo 2": plot_type2,
                #     "Gráfico tipo 3": plot_type3,
                # }

                # # Barra lateral (sidebar)
                # sidebar_selection = st.sidebar.selectbox(
                #     "Seleccione el tipo de gráfico", list(plot_options.keys())
                # )

                # # Ejecutar función según la opción seleccionada
                # plot_options[sidebar_selection]()

                # Barra lateral (sidebar)

                # self.df_top_c = top_df_final(
                #     self.df,
                #     self.main_column,
                #     self.main_num_col,
                #     self.main_cat_col,
                #     self.ascen,
                # )

                # with st.sidebar:
                #     st.title("Seleccionar gráfico")
                #     option = st.selectbox(
                #         "Seleccione el tipo de gráfico",
                #         ("Barra Vertical", "Gráfico de Pastel"),
                #     )
                #     barv_df = None  # Tu DataFrame de datos
                #     pie_df = None  # Tu DataFrame de datos

                # if option == "Barra Vertical":
                #     fig = barv_plotly(
                #         self.df_top_c,
                #         self.df_top_c.columns[0],  # Segunda columna
                #         self.df_top_c.columns[1],  # Tercera columna
                #         self.color,
                #     )

                # elif option == "Gráfico de Pastel":
                #     fig = pie_graph(
                #         self.df_top_c,
                #         self.df_top_c.columns[0],  # Segunda columna
                #         self.df_top_c.columns[1],
                #         self.color,
                #     )

            st.markdown(
                "<p style='text-align: center; font-family: Georgia, serif;'>Filtrar by {}</p>".format(
                    self.main_date_col.replace("_", " ")
                ),
                unsafe_allow_html=True,
            )
            self.fecha_min, self.fecha_max = st.select_slider(
                " ",
                options=self.df[self.main_date_col].sort_values().unique(),
                value=(
                    self.df[self.main_date_col].min(),
                    self.df[self.main_date_col].max(),
                ),
            )

            self.df_top_l = top_df_simple(
                self.df,
                self.main_date_col,
                self.main_num_col,
                self.ascen,
            )

            self.df_filtered_top_date = self.df_top_l[
                (self.df_top_l[self.df_top_l.columns[0]] >= self.fecha_min)
                & (self.df_top_l[self.df_top_l.columns[0]] <= self.fecha_max)
            ]

            self.filtro = list(self.df[self.main_column])
            self.df_top_d = (
                self.df[
                    (self.df[self.main_column].isin(self.filtro))
                    & (self.df[self.main_date_col] >= self.fecha_min)
                    & (self.df[self.main_date_col] <= self.fecha_max)
                ]
                .groupby([self.main_column, self.main_date_col], as_index=False)[
                    self.main_num_col
                ]
                .sum()
            )

            self.df_top_d.sort_values(
                self.main_num_col, ascending=self.ascen, inplace=True
            )

            st.markdown(
                "<p style='text-align: center; font-family: Georgia, serif;'>Filtrar by {}</p>".format(
                    self.main_num_col.replace("_", " ")
                ),
                unsafe_allow_html=True,
            )

            self.total_df = self.df.groupby(self.main_column, as_index=False)[
                self.main_num_col
            ].sum()

            self.total_df_date = self.df.groupby(
                [self.main_column, self.main_date_col], as_index=False
            )[self.main_num_col].sum()

            col1, col2 = st.columns(2)

            with col1:
                self.total_min = self.total_df[self.main_num_col].min()

                self.total_min = st.number_input("Mínimun Total", value=0, step=1)
            with col2:
                self.total_max = self.total_df[self.main_num_col].max()
                self.total_max = st.number_input(
                    "Maximun Total", value=self.total_max, step=1
                )

            # st.markdown(
            #     "<p style='text-align: center; font-family: Georgia, serif;'>Filtrar por media {}</p>".format(
            #         self.main_num_col.replace("_", " ")
            #     ),
            #     unsafe_allow_html=True,
            # )

            self.mean_df = (
                self.df.groupby(self.main_column, as_index=False)[self.main_num_col]
                .mean()
                .round(1)
            )

            self.mean_df_date = (
                self.df.groupby([self.main_column, self.main_date_col], as_index=False)[
                    self.main_num_col
                ]
                .mean()
                .round(1)
            )

            col1, col2 = st.columns(2)

            with col1:
                self.mean_min = self.mean_df[self.main_num_col].min()

                self.mean_min = st.number_input("Mínimun Mean", value=0, step=1)
            with col2:
                self.mean_max = self.mean_df[self.main_num_col].max()
                self.mean_max = st.number_input(
                    "Maximun Mean", value=self.mean_max, step=1.0
                )
            # self.mean_min, self.mean_max = st.select_slider(
            #     " ",
            #     options=self.mean_df[self.main_num_col].sort_values().unique(),
            #     value=(
            #         self.mean_df[self.main_num_col].min(),
            #         self.mean_df[self.main_num_col].max(),
            #     ),
            # )

            self.df_top_n = top_df_simple(
                self.df, self.main_column, self.main_num_col, self.ascen
            )

            self.df_filtered_top_num = self.df_top_n[
                (self.df_top_n[self.df_top_n.columns[1]] >= self.total_min)
                & (self.df_top_n[self.df_top_n.columns[1]] <= self.total_max)
            ]

            self.col1, self.col2 = st.columns(2)

            # Contenido de la primera columna (col1)
            with self.col1:
                # st.markdown(
                #     "<h2 style='text-align: center;'>Análisis de Variables Categoricas</h2>",
                #     unsafe_allow_html=True,
                # )

                with st.sidebar:
                    option = st.selectbox(
                        "Categorical graphic",
                        ("Bar", "Pie"),
                    )

                if option == "Bar":
                    fig = barv_plotly(
                        self.df_filtered_top_num.head(15),
                        self.df_filtered_top_num.columns[0],  # Segunda columna
                        self.df_filtered_top_num.columns[1],  # Tercera columna
                        self.color,
                    )
                    st.plotly_chart(fig)

                elif option == "Pie":
                    self.df_top_s = top_df_simple(
                        self.df, self.main_column, self.main_num_col, self.ascen
                    )

                    fig = pie_graph(
                        self.df_filtered_top_num.head(15),
                        self.df_filtered_top_num.columns[0],  # Segunda columna
                        self.df_filtered_top_num.columns[1],
                        self.color,
                    )

                    st.plotly_chart(fig)

                # st.markdown(
                #     "<h2 style='text-align: center;'>Análisis de Variables Temporales</h2>",
                #     unsafe_allow_html=True,
                # )

            with self.col2:
                with st.sidebar:
                    option = st.selectbox(
                        "Datetime graphic",
                        ("Line", "Multiple-Line"),
                    )

                if option == "Line":
                    # self.df_top_l = top_df_simple(
                    #     self.df,
                    #     self.main_date_col,
                    #     self.main_num_col,
                    #     self.ascen,
                    # )
                    fig = line_graph(
                        self.df_filtered_top_date,
                        self.df_filtered_top_date.columns[0],
                        self.df_filtered_top_date.columns[1],
                    )
                    # fig = barv_plotly(
                    #     self.df_top_c,
                    #     self.df_top_c.columns[0],  # Segunda columna
                    #     self.df_top_c.columns[1],  # Tercera columna
                    #     self.color,
                    # )
                    st.plotly_chart(fig)

                elif option == "Multiple-Line":
                    # top_df_simple(
                    #     self.df,
                    #     self.main_cat_col,
                    #     self.main_num_col,
                    #     self.main_date_col,
                    # )

                    fig = line_graph_mult(
                        self.df_top_d,
                        self.df_top_d.columns[1],
                        self.df_top_d.columns[2],
                        self.df_top_d.columns[0],
                    )

                    st.plotly_chart(fig)

            self.df_filtered = self.total_df[
                (self.total_df[self.main_num_col] >= self.total_min)
                & (self.total_df[self.main_num_col] <= self.total_max)
            ]

            self.df_filtered = self.df_filtered.sort_values(
                by=self.main_num_col, ascending=False
            )

            self.df_filtered_date = self.total_df_date[
                (self.total_df_date[self.main_num_col] >= self.total_min)
                & (self.total_df_date[self.main_num_col] <= self.total_max)
                & (self.total_df_date[self.main_date_col] >= self.fecha_min)
                & (self.total_df_date[self.main_date_col] <= self.fecha_max)
            ]

            self.df_filtered_date = self.df_filtered_date.sort_values(
                by=self.main_num_col, ascending=False
            )

            unique_categoricas_tot = (
                self.df_filtered[[self.main_column, self.main_num_col]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            unique_categoricas_tot.columns = [self.main_column, "total"]

            unique_categoricas_date_tot = (
                self.df_filtered_date[
                    [self.main_column, self.main_num_col, self.main_date_col]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            unique_categoricas_date_tot.columns = [
                self.main_column,
                "total",
                self.main_date_col,
            ]

            self.df_filtered = self.mean_df[
                (self.mean_df[self.main_num_col] >= self.mean_min)
                & (self.mean_df[self.main_num_col] <= self.mean_max)
            ]

            self.df_filtered_date = self.mean_df_date[
                (self.mean_df_date[self.main_num_col] >= self.mean_min)
                & (self.mean_df_date[self.main_num_col] <= self.mean_max)
                & (self.mean_df_date[self.main_date_col] >= self.fecha_min)
                & (self.mean_df_date[self.main_date_col] <= self.fecha_max)
            ]

            self.df_filtered = self.df_filtered.sort_values(
                by=self.main_num_col, ascending=False
            )

            self.df_filtered_date = self.df_filtered_date.sort_values(
                by=self.main_num_col, ascending=False
            )

            unique_categoricas = (
                self.df_filtered[[self.main_column, self.main_num_col]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            unique_categoricas.columns = [self.main_column, "mean"]

            # unique_categoricas.columns = [self.main_column, "mean"]

            unique_categoricas_date = (
                self.df_filtered_date[
                    [self.main_column, self.main_num_col, self.main_date_col]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            unique_categoricas_date.columns = [
                self.main_column,
                "mean",
                self.main_date_col,
            ]

            # unique_categoricas_date = [
            #     self.main_column,
            #     "mean",
            #     self.main_date_col,
            # ]

            col1, col2, col3, col4, col5 = st.columns(5)

            # Contenido de la primera columna (col1)

            with col1:
                # colu1, colu2 = st.columns(2)
                # with colu1:
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} total</p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} results found</p>".format(
                        len(unique_categoricas_tot)
                    ),
                    unsafe_allow_html=True,
                )

                st.dataframe(unique_categoricas_tot)
            with col3:
                # with colu2:
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} mean</p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} results</p>".format(
                        len(unique_categoricas)
                    ),
                    unsafe_allow_html=True,
                )
                st.dataframe(unique_categoricas)

            with col2:
                # colu1, colu2 = st.columns(2)
                # with colu1:
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} total by {}</p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                        self.main_date_col.replace("_", " "),
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} results</p>".format(
                        len(unique_categoricas_date_tot)
                    ),
                    unsafe_allow_html=True,
                )

                st.dataframe(unique_categoricas_date_tot)

            with col4:
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} mean by {}</p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                        self.main_date_col.replace("_", " "),
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>{} results</p>".format(
                        len(unique_categoricas_date)
                    ),
                    unsafe_allow_html=True,
                )
                st.dataframe(unique_categoricas_date)
            with col5:
                st.markdown(
                    "<p style='text-align: center; font-family: Arial;'>Advanced queries of {} and {}</p>".format(
                        self.main_column.replace("_", " ").capitalize(),
                        self.main_cat_col.replace("_", " "),
                    ),
                    unsafe_allow_html=True,
                )
                self.unique_main_column_values = self.df[self.main_column].unique()
                self.selected_main_column = st.selectbox(
                    self.main_column.replace("_", " ").capitalize(),
                    self.unique_main_column_values,
                )

                self.unique_main_cat_col_values = self.df[self.main_cat_col].unique()
                self.selected_main_cat_col = st.selectbox(
                    self.main_cat_col.replace("_", " ").capitalize(),
                    self.unique_main_cat_col_values,
                )

                self.df_cat = self.df.groupby(
                    [self.main_column, self.main_cat_col],
                    as_index=False,
                )[[self.main_num_col]].sum()

                self.df_cat_agg = (
                    self.df.groupby(
                        [self.main_column, self.main_cat_col],
                        as_index=False,
                    )[[self.main_num_col]]
                    .agg(["mean", "min", "max"])
                    .round(2)
                )

                self.df_cat = self.df_cat.merge(
                    self.df_cat_agg, on=[self.main_column, self.main_cat_col]
                )

                self.df_cat_filt = self.df_cat[
                    (self.df_cat[self.main_column] == self.selected_main_column)
                    & (self.df_cat[self.main_cat_col] == self.selected_main_cat_col)
                ]

                self.df_cat_filt.columns = [
                    self.main_column,
                    self.main_cat_col,
                    "total",
                    "mean",
                    "min",
                    "max",
                ]
                self.df_cat_filt = self.df_cat_filt.T

                self.df_cat_filt.columns = ["Information"]
                st.dataframe(self.df_cat_filt)

    def run(self, url):
        self.url = url

        # Función principal para ejecutar el análisis de Business Intelligence
        # self.add_bg_from_local("fo.jpg")
        self.add_bg_color()
        self.load_file()
        self.visualize_data()
        # self.generate_insights()
        # self.build_prediction_models()
        # self.generate_conclusions()


if __name__ == "__main__":
    bi = EDA()
    url = (
        "https://www.giant.com.my/our-history/"  # La URL que deseas pasar a la función
    )
    bi.run(url)


# # en terminal poner dirección relativa de carpeta de report.py
# # py -m streamlit run preprocesamiento.py
