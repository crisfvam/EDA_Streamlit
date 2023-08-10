###----------------------------------------------Import Libraries-------------------------------------------###

# Librerías para manipulación y análisis de datos
import pandas as pd

# Librería para la manipulación de archivos pickle
import pickle

# Librería para crear aplicaciones web interactivas
import streamlit as st

# Librerías para visualización de datos
import plotly.graph_objects as go

# Funciones personalizadas
from tools.my_functions import (
    modify_data_types,
    transformar_columnas_datetime,
    get_column_types,
    barv_plotly,
    pie_graph,
    line_graph,
    top_df_simple,
    line_graph_mult,
    mul_bar,
)

pd.options.display.max_columns = None
st.set_page_config(
    page_title="EDA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# st.set_option("deprecation.showPyplotGlobalUse", False)
# advertencia en matplotlib


class EDA:
    ###--------------------Variables antes de cargar el dataset---------------------###
    def __init__(self):
        self.df = None
        self.separator = ","
        self.bg_image = None

    ###--------------------Variables antes de cargar el dataset---------------------###
    # def settings_design_app(self):

    def load_data(self):
        uploaded_file = st.file_uploader(
            "Cargar archivo", type=["xlsx", "csv", "pickle"]
        )
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == "xlsx":
                self.dataset = pd.read_excel(uploaded_file)
            elif file_extension == "csv":
                if uploaded_file.size == 0:
                    self.dataset = None
                else:
                    self.dataset = pd.read_csv(uploaded_file, sep=self.separator)
            elif file_extension == "pickle":
                self.dataset = pickle.load(uploaded_file)

            #########-----------------------------------------------Filtros, dataframes y variables---------------------------------------########
            option_selected = st.checkbox("Show dataframe")
            # Verificar si el checkbox está seleccionado
            if option_selected:
                st.dataframe(self.dataset)

        else:
            self.dataset = pd.read_csv("input/EDA.csv")
            option_selected = st.checkbox("Show dataframe")
            # Verificar si el checkbox está seleccionado
            if option_selected:
                st.dataframe(self.dataset)
            # Verificar si el checkbox está seleccionado

            # title_html = f"""
            # <div style="text-align: center;">
            #     <h1 style="font-family: Helvetica Neue, serif;
            #    font-style: italic;
            #    font-weight: bold;
            #    font-size: 35px;
            #    color: white;
            #    background-repeat: no-repeat;
            #    background-size: cover;
            #    padding: 20px;">Business Intelligence EDA</h1>
            # </div>

            # """

            # st.markdown(title_html, unsafe_allow_html=True)

            # st.write("---")
            option_selected = st.checkbox("information")
            if option_selected:
                st.markdown(
                    "<div align='center'><h1><b><i><font face='Arial'>Business Intelligence EDA</font></i></b></h1></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                                <style>
                                .gold-text {
                                    color: goldenrod;
                                    font-weight: bold;
                                    font-size: 1.2em;
                                }
                                .blue-text {
                                    color: lightblue;
                                    font-weight: bold;
                                }
                                .lightpurple-text {
                                    color: lavender;
                                    font-weight: bold;
                                }
                                .lightred-text {
                                    color: lightcoral;
                                    font-weight: bold;
                                }
                                </style>
                                """,
                    unsafe_allow_html=True,
                )

                st.write(
                    "<span class='blue-text'>About dataset:</span> Must have  been processed before .",
                    unsafe_allow_html=True,
                )

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
                self.dataset = None
                link_text = "Proccess Dataframe in this app"
                link_url = (
                    "https://crfvalenciam-data-cleansing-etl-r1kik7.streamlit.app/"
                )

                st.markdown(f"[{link_text}]({link_url})")

                st.write(
                    "<span class='blue-text'>Columns Type:</span> Must have columns of type datetime, object, category, float, and int.",
                    unsafe_allow_html=True,
                )

                st.write(
                    "<span class='blue-text'>Files format:</span> CSV files works better.",
                    unsafe_allow_html=True,
                )

    def settings_and_process_data(self):
        if self.dataset is not None:
            modify_data_types(self.dataset, categories_number=150)
            option_selected = st.checkbox("Type columns")
            # Verificar si el checkbox está seleccionado
            if option_selected:
                st.write(self.dataset.dtypes)
            transformar_columnas_datetime(self.dataset)

            self.df = self.dataset.copy()
            modify_data_types(self.df, categories_number=150)
            transformar_columnas_datetime(self.df)
            (
                self.date_cols,
                self.num_cols,
                self.obj_cols,
                self.cat_cols,
            ) = get_column_types(self.df)

            self.str_list = list(self.obj_cols) + list(self.cat_cols)
            self.cat_list = list(self.cat_cols) + list(self.obj_cols)
            self.df[self.num_cols] = self.df[self.num_cols].astype(float)
            self.asce_list = [False, True]
            self.top = 10
            self.orden = 1
            self.height_one = 350
            self.width_one = 600
            self.height_two = 350
            self.width_two = 600

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if "country" in self.str_list:
                    default_option = "country"
                else:
                    default_option = self.str_list[0]  # Puedes ajustar la opción predeterminada según tus necesidades

                self.main_column = st.selectbox("Categoric col #1", self.str_list, index=self.str_list.index(default_option))

            # st.write("---")

            with col2:
                # st.markdown(
                #     "<p style='text-align: center; font-family: Georgia, serif;'>Filter by Total {}</p>".format(
                #         self.main_num_col.replace("_", " ")
                #     ),
                #     unsafe_allow_html=True,
                # )

                self.main_cat_col = st.selectbox("Categoric col #2", self.cat_cols)

            with col3:
                self.ascen = st.selectbox("Top/Last", [False, True])
                self.ascend = "Top"
                if self.ascen == True:
                    self.ascend = "Last"
            with col4:
                if "sales" in self.str_list:
                    default_option = "sales"
                else:
                    default_option = self.num_cols[0]  # Puedes ajustar la opción predeterminada según tus necesidades
                self.main_num_col = st.selectbox("Numeric column", self.num_cols)

            with col5:
                if "year" in self.str_list:
                    default_option = "year"
                else:
                    default_option = self.date_cols[0]  # Puedes ajustar la opción predeterminada según tus necesidades
                self.main_date_col = st.selectbox("Datetime column", self.date_cols)

            colum1, colum2 = st.columns(2)
            with colum1:
                try:
                    self.df[self.main_date_col] = pd.to_datetime(
                        self.df[self.main_date_col], format="%d/%m/%Y"
                    )

                    self.df[self.main_date_col] = pd.to_datetime(
                        self.df[self.main_date_col],
                        format="%Y-%m-%d %H:%M:%S",
                        errors="coerce",
                    ).dt.strftime("%Y/%m/%d")

                    self.fecha_min, self.fecha_max = st.select_slider(
                        " ",
                        options=self.df[self.main_date_col]
                        .sort_values(
                            self.main_date_col.dt.year,
                            ascending=False,
                            inplace=True,
                        )
                        .unique(),
                        value=(
                            self.df[self.main_date_col].min(),
                            self.df[self.main_date_col].max(),
                        ),
                    )

                except:
                    self.fecha_min, self.fecha_max = st.select_slider(
                        " ",
                        options=self.df[self.main_date_col].sort_values().unique(),
                        value=(
                            self.df[self.main_date_col].min(),
                            self.df[self.main_date_col].max(),
                        ),
                    )

            self.total_df = self.df.groupby(self.main_column, as_index=False)[
                self.main_num_col
            ].sum()
            self.total_df_date = self.df.groupby(
                [self.main_column, self.main_date_col], as_index=False
            )[self.main_num_col].sum()

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
            with colum2:
                column1, column2, column3, column4 = st.columns(4)
                with column1:
                    self.total_min = self.total_df[self.main_num_col].min()
                    self.total_min = st.number_input(
                        "Mín Total {}".format(self.main_num_col), value=0.0, step=1.0
                    )

                with column2:
                    self.total_max = self.total_df[self.main_num_col].max()
                    self.total_max = st.number_input(
                        "Max Total {}".format(self.main_num_col),
                        value=self.total_max,
                        step=1.0,
                    )

                # st.markdown(
                #     "<p style='text-align: center; font-family: Georgia, serif;'>Filter by Mean {}</p>".format(
                #         self.main_num_col.replace("_", " ")
                #     ),
                #     unsafe_allow_html=True,
                # )

                with column3:
                    self.mean_min = self.mean_df[self.main_num_col].min()
                    self.mean_min = st.number_input(
                        "Mín-Mean {}".format(self.main_num_col), value=0.0, step=1.0
                    )

                with column4:
                    self.mean_max = self.mean_df[self.main_num_col].max()
                    self.mean_max = st.number_input(
                        "Max-mean {}".format(self.main_num_col),
                        value=self.mean_max,
                        step=1.0,
                    )

                # with col4:
                #     st.write("---")

                # Dataframes para grafico de columna datetime respecto a una variable numerica

                self.df_top_l = top_df_simple(
                    self.df,
                    self.main_date_col,
                    self.main_num_col,
                    self.ascen,
                )
                self.df_top_l = self.df_top_l.sort_values(
                    self.df_top_l.columns[0], ascending=True
                )

                # st.dataframe(self.df_top_l)

                self.df_filtered_top_date = self.df_top_l[
                    (self.df_top_l[self.df_top_l.columns[0]] >= self.fecha_min)
                    & (self.df_top_l[self.df_top_l.columns[0]] <= self.fecha_max)
                ]

                # st.dataframe(self.df_filtered_top_date)

                self.filtro_1 = list(self.df[self.main_column])
                self.df_top_d = (
                    self.df[
                        (self.df[self.main_column].isin(self.filtro_1))
                        & (self.df[self.main_date_col] >= self.fecha_min)
                        & (self.df[self.main_date_col] <= self.fecha_max)
                    ]
                    .groupby([self.main_column, self.main_date_col], as_index=False)[
                        self.main_num_col
                    ]
                    .sum()
                ).sort_values(by=self.main_date_col, ascending=False)

                # Dataframes para grafico de columna categorica respecto a una variable numerica

                self.df_top_bar = top_df_simple(
                    self.df,
                    self.main_cat_col,
                    self.main_num_col,
                    self.ascen,
                )
                self.df_top_bar = self.df_top_bar.sort_values(
                    self.df_top_bar.columns[0], ascending=False
                )

                self.filtro = list(self.df[self.main_column])
                self.df_top_bar = (
                    self.df[
                        (self.df[self.main_column].isin(self.filtro))
                        & (self.df[self.main_num_col] >= self.total_min)
                        & (self.df[self.main_num_col] <= self.total_max)
                    ]
                    .groupby([self.main_column, self.main_cat_col], as_index=False)[
                        self.main_num_col
                    ]
                    .sum()
                    .sort_values(by=self.main_num_col, ascending=False)
                )

                # Dataframes para grafico de columna datetime respecto a una variable numerica y a una categorica
                self.df_top_n = top_df_simple(
                    self.df, self.main_column, self.main_num_col, self.ascen
                )

                self.df_filtered_top_num = self.df_top_n[
                    (self.df_top_n[self.df_top_n.columns[1]] >= self.total_min)
                    & (self.df_top_n[self.df_top_n.columns[1]] <= self.total_max)
                ]

                ### Dataframe de queries avanzadas #####

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
                    by=self.main_date_col, ascending=False
                )

                self.unique_categoricas_tot = (
                    self.df_filtered[[self.main_column, self.main_num_col]]
                    .drop_duplicates()
                    .set_index([self.main_column])
                )

                self.unique_categoricas_tot.columns = ["total"]

                self.unique_categoricas_date_tot = (
                    self.df_filtered_date[
                        [self.main_column, self.main_num_col, self.main_date_col]
                    ]
                    .drop_duplicates()
                    .set_index(self.main_date_col)
                )

                self.unique_categoricas_date_tot.columns = [
                    self.main_column,
                    "total",
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

                self.unique_categoricas = (
                    self.df_filtered[[self.main_column, self.main_num_col]]
                    .drop_duplicates()
                    .set_index([self.main_column])
                )

                self.unique_categoricas.columns = ["mean"]

                self.unique_categoricas_date = (
                    self.df_filtered_date[
                        [self.main_date_col, self.main_column, self.main_num_col]
                    ]
                    .drop_duplicates()
                    .set_index(self.main_date_col)
                )

                self.unique_categoricas_date.columns = [
                    self.main_column,
                    "mean",
                ]

        ###-----------------------------------------------Mostrar datos en la app-----------------------#####

    def visualize_data(self):
        if self.dataset is not None:
            # st.write("---")

            self.col1, self.col2 = st.columns(2)

            # Contenido de la primera columna (col1)
            with self.col1:
                colu1, colu2 = st.columns(2)

                with colu2:
                    self.color = st.selectbox(
                        " Graphic #1 color",
                        ["Inferno", "Turbo", "Emrld", "Plasma", "Magma", "Viridis"],
                        index=0,
                    )
                with colu1:
                    option = st.selectbox(
                        "Categorical graphic",
                        ("Bar", "Pie", "Stacked Bars"),
                    )

                if option == "Bar":
                    fig = barv_plotly(
                        self.df_filtered_top_num.head(15),
                        self.df_filtered_top_num.columns[0],  # Segunda columna
                        self.df_filtered_top_num.columns[1],  # Tercera columna
                        self.color,
                        self.height_one,
                        self.width_one,
                    )
                    st.plotly_chart(fig)

                elif option == "Pie":
                    fig = pie_graph(
                        self.df_filtered_top_num.head(15),
                        self.df_filtered_top_num.columns[0],  # Segunda columna
                        self.df_filtered_top_num.columns[1],
                        self.color,
                        self.height_one,
                        self.width_one,
                    )

                    st.plotly_chart(fig)

                elif option == "Stacked Bars":
                    fig = mul_bar(
                        self.df_top_bar.head(15),
                        self.df_top_bar.columns[0],
                        self.df_top_bar.columns[1],
                        self.df_top_bar.columns[2],
                    )

                    st.plotly_chart(fig)

                    # Lanzar error personalizado

            # st.markdown(
            #     "<h2 style='text-align: center;'>Análisis de Variables Temporales</h2>",
            #     unsafe_allow_html=True,
            # )

            with self.col2:
                colum1, colum2 = st.columns(2)
                with colum1:
                    option = st.selectbox(
                        "Datetime graphic",
                        ("Line", "Multiple-Line"),
                    )
                # st.dataframe(self.df_filtered_top_date)
                if option == "Line":
                    fig = line_graph(
                        self.df_filtered_top_date,
                        self.df_filtered_top_date.columns[0],
                        self.df_filtered_top_date.columns[1],
                        self.height_two,
                        self.width_two,
                    )

                    st.plotly_chart(fig)

                elif option == "Multiple-Line":
                    fig = line_graph_mult(
                        self.df_top_d,
                        self.df_top_d.columns[1],
                        self.df_top_d.columns[2],
                        self.df_top_d.columns[0],
                        self.height_two,
                        self.width_two,
                    )

                    st.plotly_chart(fig)

                with colum2:
                    self.color = st.selectbox(
                        "Graphic #2 color",
                        ["Emrld", "Turbo", "Inferno", "Plasma", "Magma", "Viridis"],
                        index=0,
                    )

            st.write("---")

            title_html = f"""
            <div style="text-align: center;">
                <h1 style="font-family: Helvetica Neue, serif;
               font-style: italic;
               font-weight: bold;
               font-size: 35px;
               color: white;
               background-repeat: no-repeat;
               background-size: cover;
               padding: 20px;">Advanced Queries</h1>
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

            col1, col2, col3, col4, col5 = st.columns(5)

            # Contenido de la primera columna (col1)

            with col1:
                # colu1, colu2 = st.columns(2)
                # with colu1:
                st.markdown(
                    "<p style= font-family: Arial;'>{} total : {} results </p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                        len(self.unique_categoricas_tot),
                    ),
                    unsafe_allow_html=True,
                )

                st.dataframe(self.unique_categoricas_tot)
            with col3:
                # with colu2:
                st.markdown(
                    "<p style= font-family: Arial;'>{} mean : {} results</p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                        len(self.unique_categoricas),
                    ),
                    unsafe_allow_html=True,
                )

                st.dataframe(self.unique_categoricas)

            with col2:
                # colu1, colu2 = st.columns(2)
                # with colu1:
                st.markdown(
                    "<p style= font-family: Arial;'>{} total by {} : {} results</p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                        self.main_date_col.replace("_", " "),
                        len(self.unique_categoricas_date_tot),
                    ),
                    unsafe_allow_html=True,
                )
                # st.markdown(
                #     "<p style='text-align: center; font-family: Arial;'>{} results</p>".format(

                #     ),
                #     unsafe_allow_html=True,
                # )

                st.dataframe(self.unique_categoricas_date_tot)

            with col4:
                st.markdown(
                    "<p style= font-family: Arial;'>{} mean by {} : {} results </p>".format(
                        self.main_num_col.replace("_", " ").capitalize(),
                        self.main_date_col.replace("_", " "),
                        len(self.unique_categoricas_date),
                    ),
                    unsafe_allow_html=True,
                )

                st.dataframe(self.unique_categoricas_date)
            with col5:
                st.markdown(
                    "<p style= font-family: Arial;'>Query of {} and {}</p>".format(
                        self.main_column.replace("_", " ").capitalize(),
                        self.main_cat_col.replace("_", " "),
                    ),
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    self.unique_main_column_values = self.df[self.main_column].unique()
                    self.selected_main_column = st.selectbox(
                        self.main_column.replace("_", " ").capitalize(),
                        self.unique_main_column_values,
                    )
                with col2:
                    self.unique_main_cat_col_values = self.df[
                        self.main_cat_col
                    ].unique()
                    self.selected_main_cat_col = st.selectbox(
                        self.main_cat_col.replace("_", " ").capitalize(),
                        self.unique_main_cat_col_values,
                    )

                self.df_obj = (
                    self.df.groupby(self.main_column, as_index=False)[self.main_num_col]
                    .agg(["sum", "mean", "max", "min"])
                    .round(1)
                )

                self.df_obj.columns = ["total", "mean", "min", "max"]
                self.df_obj = self.df_obj.reset_index()

                self.df_obj_filt = self.df_obj[
                    (self.df_obj[self.main_column] == self.selected_main_column)
                ].T

                self.df_obj_filt.columns = ["Query #1"]

                self.df_cat = (
                    self.df.groupby(self.main_cat_col, as_index=False)[
                        self.main_num_col
                    ]
                    .agg(["sum", "mean", "max", "min"])
                    .round(1)
                )

                self.df_cat.columns = ["total", "mean", "min", "max"]
                self.df_cat = self.df_cat.reset_index()

                self.df_cat_filt = self.df_cat[
                    (self.df_cat[self.main_cat_col] == self.selected_main_cat_col)
                ].T

                self.df_cat_filt.columns = ["Query #2"]

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(self.df_obj_filt)
                with col2:
                    st.dataframe(self.df_cat_filt)

                # self.df_cat_filt = self.df_cat_filt.T
                # st.dataframe(self.df_obj_filt.T)
                # self.df_cat_filt.columns = ["Information"]
                # st.dataframe(self.df_cat_filt)
                self.df_query = (
                    self.df.groupby(
                        [self.main_column, self.main_cat_col], as_index=False
                    )[self.main_num_col]
                    .agg(["sum", "mean", "max", "min"])
                    .round(1)
                )

                # st.dataframe(self.df_query)

                self.df_query.columns = ["total", "mean", "min", "max"]
                self.df_query = self.df_query.reset_index()

                self.df_query_filt = self.df_query[
                    (self.df_query[self.main_column] == self.selected_main_column)
                    & (self.df_query[self.main_cat_col] == self.selected_main_cat_col)
                ]

                # self.df_query_filt.columns = [" Query # 3"]

                st.dataframe(self.df_query_filt)

    def run(self):
        # self.url = url

        # self.settings_design_app()
        self.load_data()
        self.settings_and_process_data()
        self.visualize_data()


if __name__ == "__main__":
    app = EDA()
    app.run()
    # url = (
    #     "https://www.giant.com.my/our-history/"  # La URL que deseas pasar a la función
    # )
    # app.run(url)


# # en terminal poner dirección relativa de carpeta de report.py
# # py -m streamlit run preprocesamiento.py
