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
    get_column_types,
    top_df,
    barv_plotly,
    pie_graph,
    sec_bar_mdf,
    line_graph,
    top_df_simple,
    line_graph_mult,
)


pd.options.display.max_columns = None
st.set_page_config(
    page_title="EDA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


class BusinessIntelligence:
    def __init__(self):
        self.df = None
        self.separator = ","
        self.bg_image = None
        self.bg_image_url = (
            "https://img.freepik.com/foto-gratis/fondo-verde-abstracto_1017-3218.jpg"
        )

    # def add_bg_from_local(self, image_file):
    #     with open(image_file, "rb") as file:
    #         encoded_string = base64.b64encode(file.read())
    #     st.markdown(
    #         f"""
    #         <style>
    #             .stApp {{
    #             background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
    #             background-size: cover
    #         }}
    #         </style>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    def add_bg_from_url(self, bg_image_url):
        response = requests.get(bg_image_url)
        bg_image = response.content
        bg_image_base64 = base64.b64encode(bg_image).decode()
        title_html = f"""
            <div style="text-align: center;">
                <img src="https://upload.wikimedia.org/wikipedia/en/c/c9/Giant_logo.png" alt="Logo" style="width: 150px; height: auto;">
                <h1 style="font-family: Helvetica Neue, serif;
               font-style: italic;
               font-weight: bold;
               font-size: 35px;
               color: white;
               background-image: url(data:image/jpeg;base64,{bg_image_base64});
               background-repeat: no-repeat;
               background-size: cover;
               padding: 20px;">Business Intelligence</h1>
            </div>

        """
        st.markdown(title_html, unsafe_allow_html=True)

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
            options = {
                "Contexto": "Contexto 📊",
                "EDA": "EDA 🔍",
                "Analisis Estadistico": "Analisis Estadistico 📈",
                "Insights": "Insights ✅",
            }

            # Agrega la lista desplegable con los emojis
            selected_option = st.selectbox(
                "Selecciona una opción",
                list(options.keys()),
                format_func=lambda x: options[x],
            )
            if selected_option == "Contexto":
                # Establecer el estilo de letra más grande
                st.markdown(
                    "<style>h1 {font-size: 40px !important;}</style>",
                    unsafe_allow_html=True,
                )

                # Cambiar el estilo de letra a una más elegante
                st.markdown(
                    "<style>h1 {font-family: 'Garamond', serif;}</style>",
                    unsafe_allow_html=True,
                )

                # Cambiar el estilo de letra a una más elegante para el texto extraído
                st.markdown(
                    "<style>.extracted-text {font-family: 'Garamond', serif;}</style>",
                    unsafe_allow_html=True,
                )

                # Crear el título con el estilo de letra y formato deseado
                titulo = "<h1 style='text-align: center; color: #88ff88; font-style: italic;'>Giant Supermarket</h1>"
                st.markdown(titulo, unsafe_allow_html=True)

                # Resto de tu código...
                url = (
                    self.url
                )  # Asumiendo que la URL está almacenada en el atributo "url" de la clase
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")
                elements = soup.select(
                    ".title-after_title"
                )  # Aplicar el selector .title-after_title

                extracted_text = [element.get_text(strip=True) for element in elements]

                # Aplicar el estilo de letra y formato deseado al texto extraído
                styled_text = (
                    "<p class='extracted-text' style='font-size: 20px;'>"
                    + "</p><p class='extracted-text' style='font-size: 20px;'>".join(
                        extracted_text
                    )
                    + "</p>"
                )
                st.markdown(styled_text, unsafe_allow_html=True)

                st.dataframe(self.dataset)

            if selected_option == "EDA":
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
                    self.main_column = st.selectbox(
                        "Seleccione la variable a consultar", self.str_list
                    )
                    self.main_num_col = st.selectbox(
                        "Seleccione la variable numerica", self.num_cols
                    )
                    self.main_cat_col = st.selectbox(
                        "Seleccione la variable categorica", self.cat_cols
                    )

                    self.main_date_col = st.selectbox(
                        "Seleccione la variable fecha", self.date_cols
                    )
                    self.top = 10
                    self.orden = 1
                    # e el orden del dataframe", list(range(1, 7)))
                    self.ascen = st.selectbox("Top/Last", [False, True])
                    self.ascend = "Top"
                    if self.ascen == True:
                        self.ascend = "Last"

                    self.size_title = 2
                    self.color = st.selectbox(
                        "Color del título",
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

                    self.df_top_c = top_df(
                        self.df,
                        self.main_column,
                        self.main_num_col,
                        self.main_cat_col,
                        self.ascen,
                    )

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

                self.col1, self.col2 = st.columns(2)

                # Contenido de la primera columna (col1)
                with self.col1:
                    st.markdown(
                        "<h2 style='text-align: center;'>Análisis de Variables Categoricas</h2>",
                        unsafe_allow_html=True,
                    )

                    with st.sidebar:
                        option = st.selectbox(
                            "Graficas variables categoricas",
                            ("Barra Vertical", "Gráfico de Pastel"),
                        )
                        barv_df = None  # Tu DataFrame de datos
                        pie_df = None  # Tu DataFrame de datos

                    if option == "Barra Vertical":
                        self.df_top_s = top_df_simple(
                            self.df, self.main_column, self.main_num_col, self.ascen
                        )

                        fig = barv_plotly(
                            self.df_top_s.head(10),
                            self.df_top_s.columns[0],  # Segunda columna
                            self.df_top_s.columns[1],  # Tercera columna
                            self.color,
                        )
                        st.plotly_chart(fig)

                    elif option == "Gráfico de Pastel":
                        self.df_top_s = top_df_simple(
                            self.df, self.main_column, self.main_num_col, self.ascen
                        )

                        fig = pie_graph(
                            self.df_top_s,
                            self.df_top_s.columns[0],  # Segunda columna
                            self.df_top_s.columns[1],
                            self.color,
                        )

                        st.plotly_chart(fig)

                with self.col2:
                    st.markdown(
                        "<h2 style='text-align: center;'>Análisis de Variables Temporales</h2>",
                        unsafe_allow_html=True,
                    )

                    with st.sidebar:
                        option = st.selectbox(
                            "Graficas variables temporales",
                            ("Grafico de linea", "Grafico de linea multiple"),
                        )

                    if option == "Grafico de linea":
                        self.df_top_l = top_df_simple(
                            self.df,
                            self.main_date_col,
                            self.main_num_col,
                            self.ascen,
                        )
                        fig = line_graph(
                            self.df_top_l.head(15),
                            self.df_top_l.columns[0],
                            self.df_top_l.columns[1],
                        )
                        # fig = barv_plotly(
                        #     self.df_top_c,
                        #     self.df_top_c.columns[0],  # Segunda columna
                        #     self.df_top_c.columns[1],  # Tercera columna
                        #     self.color,
                        # )
                        st.plotly_chart(fig)

                    elif option == "Grafico de linea multiple":
                        # top_df_simple(
                        #     self.df,
                        #     self.main_cat_col,
                        #     self.main_num_col,
                        #     self.main_date_col,
                        # )

                        self.filtro = list(self.df[self.main_column])
                        self.df_top_d = (
                            self.df[self.df[self.main_column].isin(self.filtro)]
                            .groupby(
                                [self.main_column, self.main_date_col], as_index=False
                            )[self.main_num_col]
                            .sum()
                        )

                        self.df_top_d.sort_values(
                            self.main_num_col, ascending=self.ascen, inplace=True
                        )

                        fig = line_graph_mult(
                            self.df_top_d,
                            self.df_top_d.columns[1],
                            self.df_top_d.columns[2],
                            self.df_top_d.columns[0],
                        )

                        st.plotly_chart(fig)

                # self.df_top_c = top_df(
                #     self.df,
                #     self.main_column,
                #     self.main_num_col,
                #     self.main_cat_col,
                #     self.top,
                #     self.ascen,
                # )

                # fig = barv_plotly(
                #     self.df_top_c,
                #     self.df_top_c.columns[0],  # Segunda columna
                #     self.df_top_c.columns[1],  # Tercera columna
                #     self.color,
                # )
                # st.plotly_chart(fig)

                title = f"<h3 style='text-align: center; font-family: Arial, sans-serif;'><b>{self.ascen.capitalize()} {self.main_column.replace('_', ' ')} with most {self.main_num_col.replace('_', ' ')} per {self.main_cat_col.replace('_', ' ')}</b></h3>"
                st.markdown(title, unsafe_allow_html=True)
                st.write(self.df_top_c.shape)
                st.dataframe(self.df_top_c)

                # self.df_top_d = top_df_simple(
                #     self.df,
                #     self.main_cat_col,
                #     self.main_date_col,
                #     self.main_num_col,
                # )

                self.filtro = list(self.df[self.main_cat_col])
                self.df_top_d = (
                    self.df[self.df[self.main_cat_col].isin(self.filtro)]
                    .groupby([self.main_cat_col, self.main_date_col], as_index=False)[
                        self.main_num_col
                    ]
                    .sum()
                )

                st.dataframe(self.df_top_d)

                # self.col1, self.col2 = st.columns(2)

                # # Contenido de la primera columna (col1)
                # with self.col1:

                # # Contenido de la segunda columna (col2)
                # with self.col2:
                #     title = f"<h3 style='text-align: center; font-family: Arial, sans-serif;'><b>{self.ascend.replace('_', ' ').capitalize()} {self.main_date_col.replace('_', ' ')} with most {self.main_num_col.replace('_', ' ')} per {self.main_cat_col.replace('_', ' ')}</b></h3>"
                #     st.markdown(title, unsafe_allow_html=True)
                #     # title = f"<h3 style='text-align: center; font-family: Arial, sans-serif;'><b>Dataframe</b></h3>"
                #     # st.markdown(title, unsafe_allow_html=True)
                #     st.dataframe(self.df_top_d)

    # Generar insights

    # def build_prediction_models(self):
    #     st.write("---")
    #     # Construir modelos de predicción

    # def generate_conclusions(self):
    #     st.write("---")
    #     # Generar conclusiones

    def run(self, url):
        self.url = url

        # Función principal para ejecutar el análisis de Business Intelligence
        # self.add_bg_from_local("fo.jpg")
        self.add_bg_from_url(
            "https://img.freepik.com/foto-gratis/fondo-verde-abstracto_1017-3218.jpg"
        )
        self.load_file()
        self.visualize_data()
        # self.generate_insights()
        # self.build_prediction_models()
        # self.generate_conclusions()


if __name__ == "__main__":
    bi = BusinessIntelligence()
    url = (
        "https://www.giant.com.my/our-history/"  # La URL que deseas pasar a la función
    )
    bi.run(url)


# # en terminal poner dirección relativa de carpeta de report.py
# # py -m streamlit run preprocesamiento.py

# # Configurar página

# # Establecer el tema personalizado

# st.set_page_config(
#     page_title="Business Intelligence",
#     page_icon=":bar_chart:",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.set_option("deprecation.showPyplotGlobalUse", False)


# @st.cache_resource
# def cargar_datos():
#     # Cargar datos costosos aquí
#     datos = ...
#     return datos


# # Llamada a la función cargar_datos
# datos = cargar_datos()


# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#         f"""
#         <style>
#             .stApp {{
#             background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#             background-size: cover
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )


# add_bg_from_local("fo.jpg")

# # Obtener el contenido de la imagen de fondo desde la URL
# bg_image_url = "https://img.freepik.com/foto-gratis/fondo-verde-abstracto_1017-3218.jpg"
# response = requests.get(bg_image_url)
# bg_image = response.content

# # Convertir la imagen a base64
# bg_image_base64 = base64.b64encode(bg_image).decode()

# # Generar el código HTML con el título y el fondo de imagen
# title_html = f"""
#     <div style="background-image: url(data:image/jpeg;base64,{bg_image_base64});
#                 background-repeat: no-repeat;
#                 background-size: cover;
#                 padding: -0px -0px;
#                 text-align: center;">
#         <img src="https://upload.wikimedia.org/wikipedia/en/c/c9/Giant_logo.png" alt="Logo" style="width: 190px; height: auto;">
#         <h1 style="font-family: Helvetica Neue, serif;
#                 font-style: italic;
#                 font-weight: bold;
#                 font-size: 35px;
#                 color: white">Business Intelligence</h1>
#     </div>
# """

# # Mostrar el título en Streamlit
# st.markdown(title_html, unsafe_allow_html=True)

# def load_data(file, separator):
#     file_extension = file.name.split(".")[-1]

#     # Cargar el archivo en un DataFrame según su extensión
#     if file_extension == "xlsx":
#         dataset = pd.read_excel(file)
#     elif file_extension == "csv":
#         if file.size == 0:
#             dataset = None
#         # Leer el archivo CSV con el separador seleccionado
#         dataset = pd.read_csv(file, sep=separator)
#     # elif file_extension == "xml":
#     #     dataset = pd.read_xml(file)
#     # elif file_extension == "no_xml":
#     #     dataset = pd.read_csv(file)
#     else:
#         dataset = None

#     return dataset


# options = {
#     "Contexto": "🌐 Contexto",
#     "Análisis gráfico": "📊 Análisis gráfico",
#     "Insights": "✅ Insights",
#     "Modelos de predicción": "📝 Modelos de predicción",
#     "Conclusiones": "📊 Conclusiones",
# }

# # Crear el sidebar desplegable
# option = st.sidebar.selectbox("Selecciona una opción", list(options.keys()))

# if option == "Contexto":
#     st.markdown(
#         "<style>body{background-color: #008000;}</style>", unsafe_allow_html=True
#     )
#     st.markdown(
#         "<style>h1{color: #FFFFFF; font-family: Arial, sans-serif;}</style>",
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         "<style>p{color: #FFFFFF; font-family: Arial, sans-serif;}</style>",
#         unsafe_allow_html=True,
#     )

#     titulo = "<h1 style='text-align: center; font-family: Arial, sans-serif; color: #88ff88; font-style: italic;'>Giant Supermarket</h1>"
#     st.markdown(titulo, unsafe_allow_html=True)

#     texto = """
#     Giant es una cadena de supermercados fundada en 1944 en Kuala Lumpur. En 1974, abrieron el Teng Minimarket Centre (TMC) en Bangsar, lo que impulsó su expansión. En 1999, Dairy Farm International Holdings adquirió el 90% de la cadena, mientras que la familia Teng retuvo el resto. En 2003, la empresa matriz cambió su nombre a Dairy Farm Giant Retail Sdn Bhd. Actualmente, la compañía opera como subsidiaria de Dairy Farm International Holdings Limited y se le cambió el nombre a GCH Retail (Malasia) Sdn Bhd.

#     GCH Retail es propiedad en un 30% de Syarikat Pesaka Antah Sdn Bhd, una empresa controlada por la familia real de Negeri Sembilan, y el 70% restante perteneceContinuación del código con la indentación corregida:a DFI Mauritius Ltd.

#     En diciembre de 2011, Giant ingresó al mercado vietnamita, pero en febrero de 2018 vendieron su interés en una hipermercado en Vietnam a Auchan Retail Vietnam. Desde entonces, la compañía ha salido del negocio de supermercados e hipermercados en Vietnam.

#     En los últimos años, Giant ha enfrentado dificultades en Malasia y Singapur debido a altos costos operativos, debilitamiento del gasto del consumidor y mayor competencia. Han cerrado varias tiendas en ambos países.

#     En septiembre de 2020, Giant anunció una rebranding en Singapur, presentando un nuevo logo y características nuevas en sus tiendas. También redujeron los precios de los productos esenciales en un promedio del 20% durante seis meses. Esta rebranding se extendió a Malasia en abril de 2021, y se introdujo una sección llamada "Ringgit Zone" con productos a precios fijos de RM3 por artículo."""

#     st.markdown(texto, unsafe_allow_html=True)

# elif option == "Análisis gráfico":
#     # Opción para cargar el archivo
#     uploaded_file = st.file_uploader(
#         "Cargar archivo procesado", type=["xlsx", "csv", "pickle", "parquet"]
#     )

#     if uploaded_file is not None:
#         # Obtener el separador seleccionado si el archivo es de tipo CSV
#         separator = ","
#         if uploaded_file.type == "text/csv":
#             dataset = pd.read_csv(uploaded_file, sep=separator)

#         elif uploaded_file.type == "excel":
#             dataset = pd.read_excel(uploaded_file)

#         # Cargar el archivo pickle descargado
#         uploaded_file = st.file_uploader(
#             "Cargar archivo pickle descargado", type="pickle"
#         )
#         if uploaded_file is not None:
#             dataset = pd.read_pickle(uploaded_file)

#             # Verificar el tipo de dataset
#             if isinstance(dataset, pd.DataFrame):
#                 st.write("Archivo cargado exitosamente.")
#                 # Resto de tu código en Streamlit utilizando el DataFrame cargado
#             else:
#                 st.write("Error: El archivo no contiene un DataFrame válido.")

#         elif uploaded_file.name.endswith(".parquet"):
#             dataset = pq.read_table(uploaded_file).to_pandas()

#         df = dataset.copy()
#         df.dtypes
#         modify_data_types(df, categories_number=150)
#         df.dtypes

#         obj_cols = df.select_dtypes(include=["object"]).columns
#         cat_cols = df.select_dtypes(include=["category"]).columns
#         num_cols = df.select_dtypes(include=["int64", "float64"]).columns
#         date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
#         date_cols = date_cols.append(num_cols[-7:-1])
#         num_cols = num_cols[0:-7]
#         str_list = list(obj_cols) + list(cat_cols) + list(date_cols)
#         cat_list = list(obj_cols) + list(cat_cols) + list(date_cols)
#         asce_list = [False, True]

#         col1, col2 = st.columns(2)

#         # Contenido de la primera columna (col1)
#         with col1:
#             with st.sidebar:
#                 main_column = st.selectbox(
#                     "Seleccione la variable a consultar", str_list
#                 )
#                 main_num_col = st.selectbox("Seleccione la variable numerica", num_cols)
#                 main_cat_col = st.selectbox(
#                     "Seleccione la variable categorica", cat_cols
#                 )
#                 top = 10
#                 orden = st.selectbox(
#                     "Seleccione el orden del dataframe", list(range(1, 7))
#                 )
#                 ascen = st.selectbox("Descendente=False/Ascendente=True", [False, True])

#                 size_title = st.selectbox(
#                     "Tamaño del título", list(range(1, 6)), index=3
#                 )
#                 color = st.selectbox(
#                     "Color del título",
#                     ["Emrld", "Crimson", "Sapphire", "Gold", "Amethyst"],
#                     index=0,
#                 )
#                 height = st.selectbox(
#                     "Altura del gráfico", list(range(100, 1001, 100)), index=3
#                 )
#                 width = st.selectbox(
#                     "Anchura del gráfico", list(range(500, 2001, 100)), index=5
#                 )
#                 ori = st.selectbox(
#                     "Orientación del gráfico", ["Vertical", "Horizontal"], index=0
#                 )

#                 if ori == "Vertical":
#                     ori = "v"
#                 else:
#                     ori = "h"

#             df_top_c = top_df(df, main_column, main_num_col, main_cat_col, top, ascen)

#             title = f"<h3 style='text-align: center; font-family: Arial, sans-serif;'><b>{top} {main_column} with most {main_num_col}</b></h3>"
#             st.markdown(title, unsafe_allow_html=True)

#             fig = barv_plotly(
#                 df_top_c,
#                 main_column,
#                 main_num_col,
#                 orden,
#                 size_title,
#                 color,
#                 height,
#                 width,
#                 ori,
#             )
#             st.plotly_chart(fig)

#         # Contenido de la segunda columna (col2)
#         with col2:
#             fig = barv_plotly(
#                 df_top_c,
#                 main_column,
#                 main_num_col,
#                 orden,
#                 size_title,
#                 color,
#                 height,
#                 width,
#                 ori,
#             )
#             st.plotly_chart(fig)

# elif option == "Insights":
#     st.code(
#         """
#         # Código para los insights
#         """
#     )

# elif option == "Modelos de predicción":
#     st.code(
#         """
#         # Código para los modelos de predicción
#         """
#     )

# elif option == "Conclusiones":
#     st.code(
#         """
#         # Código para las conclusiones
#         """
#     )
