import pandas as pd
import numpy as np
import os
import streamlit as st
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


def load_data(file, separator):
    file_extension = file.name.split(".")[-1]

    # Cargar el archivo en un DataFrame según su extensión
    if file_extension == "xlsx":
        dataset = pd.read_excel(file)
    elif file_extension == "csv":
        if file.size == 0:
            dataset = None
        # Leer el archivo CSV con el separador seleccionado
        dataset = pd.read_csv(file, sep=separator)
    # elif file_extension == "xml":
    #     dataset = pd.read_xml(file)
    # elif file_extension == "no_xml":
    #     dataset = pd.read_csv(file)
    else:
        dataset = None

    return dataset


def determinar_formato_fecha(df, columna):
    formatos = pd.to_datetime(df[columna], infer_datetime_format=True).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    formato_mas_comun = formatos.value_counts().idxmax()
    return formato_mas_comun


def transformar_columnas_datetime(df):
    for columna in df.select_dtypes(include=["datetime64"]).columns.to_list():
        try:
            df[columna] = pd.to_datetime(
                df[columna], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            ).dt.strftime("%d/%m/%Y")
        except:
            pass
    return df


def clean_column_names(df):
    # Obtener los nombres de las columnas actuales
    column_names = df.columns.tolist()

    # Eliminar los espacios y convertir a minúsculas
    new_column_names = [col.strip().replace(" ", "_").lower() for col in column_names]

    # Asignar los nuevos nombres de columnas al DataFrame
    df.columns = new_column_names

    # Restaurar el tipo de datos original
    for col in df.columns:
        if col in column_names:
            original_dtype = df[col].dtype
            if original_dtype == int:
                df[col] = df[col].astype(int)
            elif original_dtype == float:
                df[col] = df[col].astype(float)
            elif original_dtype == "category":
                df[col] = df[col].astype("category")

    return df


# strip_func
def convert_data_types(df, original_dtypes):
    for col in df.columns:
        df[col] = df[col].astype(original_dtypes[col])
    return df


# Concatena varios archivos csv en un folder
def concat_csv_files(path_folder):
    # Lista de archivos
    files = os.listdir(path_folder)
    # Concatenamos todos los archivos en un única dataframe
    df_concat = pd.concat([pd.read_csv(path_folder + "/" + f) for f in files])
    # Devolvemos el dataframe
    return df_concat


def modify_data_types(df, categories_number=150):
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                if df[col].dtype not in [int, float]:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="ignore")
                    except ValueError:
                        pass

                if df[col].nunique() < categories_number:
                    df[col] = df[col].astype("category")

    # for columna in df.select_dtypes(include=["datetime64"]).columns.to_list():
    #     try:
    #         formatos = pd.to_datetime(
    #             df[columna], infer_datetime_format=True
    #         ).dt.strftime("%Y-%m-%d %H:%M:%S")
    #         formato_mas_comun = formatos.value_counts().idxmax()

    #         df[columna] = pd.to_datetime(
    #             df[columna], format=formato_mas_comun
    #         ).dt.strftime("%d-%m-%Y")
    #     except:
    #         pass
    return df


def strip_values_of_columns(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    return df


# percentage nulls
def p_nulls(df):
    per_nulls = df.isnull().sum() / len(df) * 100
    return per_nulls.sort_values(ascending=False).head(20)


def p_tot_nulls(df):
    per = (((df.isnull().sum()) / len(df)) * 100).mean().round(2)
    return per


def drop_nulls(df, max_percentage_nulls):
    columns_to_drop = []
    for col in df.columns:
        porcentaje = df[col].isnull().sum() * 100 / len(df[col])
        if porcentaje > max_percentage_nulls:
            columns_to_drop.append(col)

    df.drop(columns=columns_to_drop, inplace=True)

    return df


def impute_nulls(df, list_cat=None):
    for col in df.columns:
        porcentaje = df[col].isnull().sum() * 100 / len(df[col])

        if porcentaje > 0:
            if col in df.select_dtypes(include=["category"]).columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
            if col in df.select_dtypes(include=["object"]).columns:
                df[col].fillna("no_information_found", inplace=True)
            elif col in df.select_dtypes(include=["datetime64[ns]"]).columns:
                df[col].fillna(df[col].median(), inplace=True)

    if list_cat:
        for i in df.select_dtypes(include=["int64", "float64"]).columns:
            porcentaje = df[i].isnull().sum() * 100 / len(df[i])

            if porcentaje > 0:
                for j in range(len(list_cat)):
                    try:
                        grupo = df.groupby(list_cat[j:])
                        df[i] = grupo[i].transform(lambda x: x.fillna(x.mean()))
                    except:
                        break

    return df


def duplicados(df):
    dup_df = df.duplicated().sum()
    if dup_df > 0:
        df = df.drop_duplicates()
    num_duplicates = df.duplicated().sum()

    return df, dup_df, num_duplicates


# Create new columns year, month, day
def date_columns(df):
    datetime_columns = df.select_dtypes(include=["datetime64"]).columns.tolist()

    for column_name in datetime_columns:
        # df[column_name] = pd.to_datetime(df[column_name])  # convert column in datetime
        df.set_index(column_name, inplace=True)
        df["year_" + column_name] = df.index.year
        df["month_" + column_name] = df.index.month
        df["day_" + column_name] = df.index.day
        # df['hour'] = df.index.hour
        df.reset_index(inplace=True)

    return df


def get_column_types(df):
    modify_data_types(df, categories_number=150)
    date_list = []
    num_list = []
    object_list = []
    category_list = []

    for column_name in df.columns:
        column_type = df[column_name].dtype

        if column_type == "datetime64[ns]":
            date_list.append(column_name)
        elif (
            column_name.startswith("year")
            or column_name.startswith("month")
            or column_name.startswith("day")
            or column_name.endswith("date")
        ):
            date_list.append(column_name)
        if column_type == "int64" or column_type == "float64":
            num_list.append(column_name)
        elif column_type == "object":
            object_list.append(column_name)
        elif column_type == "category":
            category_list.append(column_name)

    return date_list, num_list, object_list, category_list

    # def get_column_types(df):
    #     date_list = []
    #     num_list = []
    #     object_list = []
    #     category_list = []

    #     for column_name in df.columns:
    #         column_type = df[column_name].dtype

    #         if column_type == "int64" or column_type == "float64":
    #             num_list.append(column_name)
    #         elif column_type == "object":
    #             object_list.append(column_name)
    #         elif column_type == "category":
    #             category_list.append(column_name)

    #         elif column_type == "datetime64[ns]":
    #             date_list.append(column_name)
    #         elif (
    #             column_name.startswith("year")
    #             or column_name.startswith("month")
    #             or column_name.startswith("day")
    #             or column_name.endswith("date")
    #         ):
    #             date_list.append(column_name)
    # elif column_type == "int64" or column_type == "float64":
    #     num_list.append(column_name)
    # elif column_type == "object":
    #     object_list.append(column_name)
    # elif column_type == "category":
    #     category_list.append(column_name)

    return date_list, num_list, object_list, category_list


# def p_total_out(df, z_score_threshold):
#     z_scores = (df - df.mean()) / df.std()

#     porcentaje_outliers = ((z_scores.abs() > z_score_threshold).mean() * 100).mean()

#     return porcentaje_outliers


def drop_outliers(df, numeric_columns_list, factor=1.5):
    df_outlier = df[numeric_columns_list]
    Q1 = df_outlier.quantile(0.25)
    Q3 = df_outlier.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outlier_mask = ((df_outlier < lower_bound) | (df_outlier > upper_bound)).any(axis=1)
    df_outlier = df[~outlier_mask]
    df_outliers = df[outlier_mask]

    return df_outlier, df_outliers


def impute_outliers_funct(df, num_list, factor=1.5, replace_with="min"):
    for i in num_list:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        if replace_with == "min":
            df.loc[df[i] < lower_bound, i] = lower_bound
        elif replace_with == "max":
            df.loc[df[i] > upper_bound, i] = upper_bound

    return df


def detect_outliers_z_score(df, threshold=1.96):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outliers_table = pd.DataFrame(columns=["Columna", "Outliers"])
    columns_with_outliers = []

    for col in numeric_cols:
        z_scores = (df[col] - np.mean(df[col])) / np.std(df[col])
        outliers = df[np.abs(z_scores) > threshold][col]
        if len(outliers) > 0:
            columns_with_outliers.append(col)
            outliers_table = outliers_table.append(
                {"Columna": col, "Outliers": outliers.tolist()},
                ignore_index=True,
            )

    if len(columns_with_outliers) > 0:
        # st.write("Columnas con outliers:")

        # st.write("Gráfico de Boxplot para columnas con outliers:")
        fig, ax = plt.subplots()
        df[columns_with_outliers].boxplot(ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    return outliers_table


def drop_outliers_z_score(df, numeric_columns_list, threshold=2.96):
    for col in numeric_columns_list:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers_z_score = z_scores > threshold

        df = df.loc[~outliers_z_score]

    return df


def impute_outliers(df, numeric_columns_list, threshold=2.96, impute_with="max"):
    for col in numeric_columns_list:
        if df[col].dtype in ["int64", "float64"]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold

            if impute_with == "max":
                impute_value = df.loc[~outliers, col].max()
            elif impute_with == "min":
                impute_value = df.loc[~outliers, col].min()
            else:
                raise ValueError("Invalid impute_with option. Choose 'max' or 'min'.")

            df.loc[outliers, col] = impute_value

    return df


def drop_outliers_iso(df, numeric_columns_list):
    from sklearn.ensemble import IsolationForest

    df_iso = df[numeric_columns_list]
    isolation_forest = IsolationForest(contamination="auto", random_state=357)
    outliers = isolation_forest.fit_predict(df_iso) != -1
    df_iso_for = df.loc[~outliers]
    df_out_iso = df.loc[outliers]
    return df_iso_for, df_out_iso


# ________________________________________________________________________________________________________________
# ________________________________________________________________________________________________________________


def top_df_simple(df, main_column, main_num_col, ascen):
    df_main_var = df.groupby(main_column, as_index=False)[main_num_col].sum()
    df_main_var.rename(columns={main_num_col: "total_" + main_num_col}, inplace=True)
    df_main_var.sort_values("total_" + main_num_col, ascending=ascen, inplace=True)
    return df_main_var


def top_df_complete(df, main_cat_col, main_num_col, main_date_col, ascen):
    filtro = list(df[main_cat_col])
    df_top_d = (
        df[df[main_cat_col].isin(filtro)]
        .groupby([main_cat_col, main_date_col], as_index=False)[main_num_col]
        .sum()
    )

    df_top_d.sort_values(main_num_col, ascending=ascen, inplace=True)

    return df_top_d


def statistic_df(df):
    df_description = df.describe().T
    columns_to_drop = ["count", "25%", "75%"]
    df_description.drop(columns=columns_to_drop, inplace=True)
    df_description.columns = ["mean", "STD", "median", "min", "max"]
    df_description = round(df_description, 1)
    return df_description


"""
self.filtro = list(self.df[self.main_cat_col])
                        self.df_top_d = (
                            self.df[self.df[self.main_cat_col].isin(self.filtro)]
                            .groupby(
                                [self.main_cat_col, self.main_date_col], as_index=False
                            )[self.main_num_col]
                            .sum()
                        )"""


"""def top_df(
    df,
    main_column,
    main_num_col,
    main_cat_col,
    ascen,
    filter_column=None,
    filter_value=None,
):
    # Apply filter if provided
    if filter_column is not None and filter_value is not None:
        df = df[df[filter_column] == filter_value]

    # Creación de primero; la columna de consulta principal y segundo la columna numerica
    df_main_var = (
        df.groupby([main_column], as_index=False)[main_num_col]
        .sum()
        .sort_values(by=main_num_col, ascending=False)
    )

    # Renombrando la columna numerica, ya que es la sumatoria total por registro de la columna de consulta principal
    df_main_var.rename({main_num_col: "total_" + main_num_col}, axis=1, inplace=True)

    # Columna que arroja los registros de la variable categorica que mas sumatoria tiene de la variable numerica por variable principal de consulta
    df_main_var[
        main_cat_col
        + "_with_most_"
        + main_num_col
        + "_of_the_"
        + main_column.split("_")[0]
    ] = [
        list(
            df.loc[(df[main_column] == i)]
            .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
            .sum()
            .sort_values(by=main_num_col, ascending=False)[main_cat_col]
        )[0]
        for i in df_main_var[main_column]
    ]

    # Columna que arroja el total de la sumatoria de la relacion entre el registro categorico de la columna anterior y  los registros de la variable principal de consulta
    df_main_var[
        "total_"
        + main_num_col
        + "_of_this_"
        + main_cat_col
        + "_by_"
        + main_column.split("_")[0]
    ] = [
        list(
            df.loc[(df[main_column] == i)]
            .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
            .sum()
            .sort_values(by=main_num_col, ascending=False)[main_num_col]
        )[0]
        for i in df_main_var[main_column]
    ]

    # Columna que arroja el promedio de la sumatoria de la relacion entre el registro categorico de la columna anterior y  los registros de la variable principal de consulta
    # df_main_var[
    #     "AVG_"
    #     + main_num_col
    #     + "_of_this_"
    #     + main_cat_col
    #     + "_by_"
    #     + main_column.split("_")[0]
    # ] = [
    #     round(
    #         (
    #             list(
    #                 df.loc[(df[main_column] == i)]
    #                 .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
    #                 .mean()
    #                 .sort_values(by=main_num_col, ascending=False)[main_num_col]
    #             )[0]
    #         ),
    #         2,
    #     )
    #     for i in df_main_var[main_column]
    # ]

    df_main_var = df_main_var.sort_values(
        by=df_main_var.columns.to_list()[1], ascending=ascen
    ).head(10)

    return df_main_var


def top_df(df, main_column, main_num_col, main_cat_col, ascen):
    # Creación de primero; la columna de consulta principal y segundo la columna numerica
    df_main_var = (
        df.groupby([main_column], as_index=False)[main_num_col]
        .sum()
        .sort_values(by=main_num_col, ascending=False)
    )

    # Renombrando la columna numerica, ya que es la sumatoria total por registro de la columna de consulta principal
    df_main_var.rename({main_num_col: "total_" + main_num_col}, axis=1, inplace=True)

    # Columna que arroja los registros de la variable categorica que mas sumatoria tiene de la variable numerica por variable principal de consulta
    df_main_var[
        main_cat_col
        + "_with_most_"
        + main_num_col
        + "_of_the_"
        + main_column.split("_")[0]
    ] = [
        list(
            df.loc[(df[main_column] == i)]
            .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
            .sum()
            .sort_values(by=main_num_col, ascending=False)[main_cat_col]
        )[0]
        for i in df_main_var[main_column]
    ]

    # Columna que arroja el total de la sumatoria de la relacion entre el registro categorico de la columna anterior y  los registros de la variable principal de consulta
    df_main_var[
        "total_"
        + main_num_col
        + "_of_this_"
        + main_cat_col
        + "_by_"
        + main_column.split("_")[0]
    ] = [
        list(
            df.loc[(df[main_column] == i)]
            .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
            .sum()
            .sort_values(by=main_num_col, ascending=False)[main_num_col]
        )[0]
        for i in df_main_var[main_column]
    ]

    # Columna que arroja el promedio de la sumatoria de la relacion entre el registro categorico de la columna anterior y  los registros de la variable principal de consulta
    df_main_var[
        "AVG_"
        + main_num_col
        + "_of_this_"
        + main_cat_col
        + "_by_"
        + main_column.split("_")[0]
    ] = [
        round(
            (
                list(
                    df.loc[(df[main_column] == i)]
                    .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
                    .mean()
                    .sort_values(by=main_num_col, ascending=False)[main_num_col]
                )[0]
            ),
            2,
        )
        for i in df_main_var[main_column]
    ]

    df_main_var = df_main_var.sort_values(
        by=df_main_var.columns.to_list()[1], ascending=ascen
    )

    return df_main_var"""


def top_df_final(df, main_column, main_num_col, main_cat_col, ascen):
    # Creación del DataFrame principal agrupado por la columna principal
    df_main_var = df.groupby(main_column, as_index=False)[main_num_col].sum()

    # Renombrar la columna numérica
    df_main_var.rename({main_num_col: "total_" + main_num_col}, axis=1, inplace=True)

    # Obtener la categoría con la suma más alta de la columna numérica para cada valor de la columna principal
    df_main_var[
        main_cat_col
        + "_with_most_"
        + main_num_col
        + "_of_the_"
        + main_column.split("_")[0]
    ] = [
        df.loc[df[main_column] == i]
        .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
        .sum()
        .sort_values(by=main_num_col, ascending=False)[main_cat_col]
        .iloc[0]
        for i in df_main_var[main_column]
    ]

    # Obtener la suma total de la relación entre la categoría anterior y los registros de la columna principal
    df_main_var[
        "total_"
        + main_num_col
        + "_of_this_"
        + main_cat_col
        + "_by_"
        + main_column.split("_")[0]
    ] = [
        df.loc[df[main_column] == i]
        .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
        .sum()
        .sort_values(by=main_num_col, ascending=False)[main_num_col]
        .iloc[0]
        for i in df_main_var[main_column]
    ]

    # Obtener el promedio de la relación entre la categoría anterior y los registros de la columna principal
    df_main_var[
        "AVG_"
        + main_num_col
        + "_of_this_"
        + main_cat_col
        + "_by_"
        + main_column.split("_")[0]
    ] = [
        round(
            df.loc[df[main_column] == i]
            .groupby([main_column, main_cat_col], as_index=False)[main_num_col]
            .mean()
            .sort_values(by=main_num_col, ascending=False)[main_num_col]
            .iloc[0],
            2,
        )
        for i in df_main_var[main_column]
    ]

    df_main_var = df_main_var.sort_values(by="total_" + main_num_col, ascending=ascen)

    return df_main_var


def normalize(df, norm_col):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    df[norm_col] = scaler.fit_transform(df[[norm_col]])

    return df


def df_graphic(df, main_column, main_num_col, top, ascending):
    # Agrupar por la columna principal y calcular estadísticas descriptivas de la columna numérica
    df_main_var = (
        df.groupby(main_column)[main_num_col]
        .agg(sum="sum", mean="mean", std="std", median="median", min="min", max="max")
        .reset_index()
    )

    # Renombrar las columnas
    df_main_var.rename(
        columns={
            main_column: main_column.lower(),
            "sum": "Total_" + main_num_col,
            "mean": "AVG",
            "std": "STD",
            "median": "Median",
            "min": "Min",
            "max": "Max",
        },
        inplace=True,
    )

    # Ordenar el DataFrame por la columna especificada y seleccionar los primeros 'top' registros
    df_main_var = df_main_var.sort_values(
        by="Total_" + main_num_col, ascending=ascending
    ).head(top)

    # Redondear los valores numéricos a 2 decimales
    df_main_var = df_main_var.round(2)

    return df_main_var


def enum_categoricas(df):
    from sklearn.preprocessing import LabelEncoder

    categorical_cols = df.select_dtypes(include=["category", "object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


# Statistics tests------------------------------------------------------------


from scipy.stats import shapiro, kstest, normaltest, anderson


def shapiro(df, columns):
    par_columns = []
    no_par_columns = []
    messages = []

    if len(df) > 50:
        messages.append(
            "Se recomienda usar el test de Kolmogorov-Smirnov para muestras con más de 50 registros."
        )
    else:
        for i in columns:
            resultado = shapiro(df[i])
            nivel_significancia = 0.05
            if resultado.pvalue > nivel_significancia:
                par_columns.append(i)
                messages.append(
                    "Los datos de la columna '{}' siguen una distribución normal.".format(
                        i
                    )
                )
            else:
                no_par_columns.append(i)
                messages.append(
                    "Los datos de la columna '{}' no siguen una distribución normal según el test de Shapiro-Wilk.".format(
                        i
                    )
                )
        if len(par_columns) == 0:
            messages.append(
                "Ninguna columna sigue una distribución normal según el test de Shapiro-Wilk."
            )

    return par_columns, no_par_columns, messages


def kolgomorov(df):
    par_columns = []
    no_par_columns = []
    messages = []

    for i in df.columns:
        resultado = kstest(df[i], cdf="norm")
        nivel_significancia = 0.05
        if resultado.pvalue > nivel_significancia:
            par_columns.append(i)
            messages.append(
                "Los datos de la columna '{}' siguen una distribución normal.".format(i)
            )
        else:
            no_par_columns.append(i)
            messages.append(
                "Los datos de la columna '{}' no siguen una distribución normal según el test de Kolmogorov-Smirnov.".format(
                    i
                )
            )
    if len(par_columns) == 0:
        messages.append(
            "Ninguna columna sigue una distribución normal según el test de Kolmogorov-Smirnov."
        )

    return par_columns, no_par_columns, messages


def d_agostino(df):
    par_columns = []
    no_par_columns = []
    messages = []

    for i in df.columns:
        resultado = normaltest(df[i])
        nivel_significancia = 0.05
        if resultado.pvalue > nivel_significancia:
            par_columns.append(i)
            messages.append(
                "Los datos de la columna '{}' siguen una distribución normal.".format(i)
            )
        else:
            no_par_columns.append(i)
            messages.append(
                "Los datos de la columna '{}' no siguen una distribución normal según el test de D'Agostino.".format(
                    i
                )
            )
    if len(par_columns) == 0:
        messages.append(
            "Ninguna columna sigue una distribución normal según el test de D'Agostino."
        )

    return par_columns, no_par_columns, messages


def anderson_darling(df, num_cols):
    from scipy.stats import anderson

    par_columns = []
    no_par_columns = []
    messages = []

    for n in num_cols:
        result = anderson(df[n])

        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < cv:
                par_columns.append(n)
                messages.append(
                    "La columna '{}' sigue una distribución Gaussiana para una significancia del {}%.".format(
                        n, sl
                    )
                )
            else:
                no_par_columns.append(n)
                messages.append(
                    "La columna '{}' no sigue una distribución Gaussiana según el test de Anderson-Darling.".format(
                        n
                    )
                )

    if len(par_columns) == 0:
        messages.append(
            "Ninguna columna sigue una distribución Gaussiana según el test de Anderson-Darling."
        )

    return par_columns, no_par_columns, messages


def correlacion_parametricas(par_columns, df):
    from scipy.stats import pearsonr

    alta_corr_par = []
    messages = []

    for i in range(len(par_columns)):
        for j in range(i + 1, len(par_columns)):
            pearson_correlation, p_value = pearsonr(
                df[par_columns[i]], df[par_columns[j]]
            )
            r_square = pearson_correlation**2
            if pearson_correlation > 0.5 and r_square > 0.5:
                significancia = (1 - p_value) * 100
                alta_corr_par.append((par_columns[i], par_columns[j], significancia))
                messages.append(
                    "Alta correlación entre las columnas '{}' y '{}'. Significancia: {:.2f}%".format(
                        par_columns[i], par_columns[j], significancia
                    )
                )

    if len(alta_corr_par) == 0:
        messages.append(
            "No se encontraron columnas con alta correlación según Pearson y R cuadrado."
        )

    return alta_corr_par, messages


def correlacion_no_parametricas(no_par_columns, df):
    from scipy.stats import spearmanr, kendalltau

    alta_corr_no_par = []
    messages = []

    for i in range(len(no_par_columns)):
        for j in range(i + 1, len(no_par_columns)):
            spearman_corr, spearman_p = spearmanr(
                df[no_par_columns[i]], df[no_par_columns[j]]
            )
            kendall_corr, kendall_p = kendalltau(
                df[no_par_columns[i]], df[no_par_columns[j]]
            )

            if spearman_corr > 0.5 and kendall_corr > 0.5:
                spearman_significance = (1 - spearman_p) * 100
                kendall_significance = (1 - kendall_p) * 100

                alta_corr_no_par.append(
                    (
                        no_par_columns[i],
                        no_par_columns[j],
                        spearman_significance,
                        kendall_significance,
                    )
                )
                messages.append(
                    "Alta correlación entre las columnas '{}' y '{}'. Significancia (Spearman): {:.2f}%, Significancia (Kendall): {:.2f}%".format(
                        no_par_columns[i],
                        no_par_columns[j],
                        spearman_significance,
                        kendall_significance,
                    )
                )

    if len(alta_corr_no_par) == 0:
        messages.append(
            "No se encontraron columnas con alta correlación según Spearman y Kendall."
        )

    return alta_corr_no_par, messages


def evaluar_dependencia(df, num_cols):
    from itertools import combinations
    from scipy.stats import ttest_ind, mannwhitneyu

    dependientes = []
    independientes = []

    for col1, col2 in combinations(num_cols, 2):
        stat, p_value = ttest_ind(df[col1], df[col2], equal_var=False)
        if p_value < 0.05:
            dependientes.append((col1, col2))
        else:
            independientes.append((col1, col2))

    for col1, col2 in combinations(num_cols, 2):
        stat, p_value = mannwhitneyu(df[col1], df[col2], alternative="two-sided")
        if p_value < 0.05:
            dependientes.append((col1, col2))
        else:
            independientes.append((col1, col2))

    message = "Variables dependientes:\n"
    for cols in dependientes:
        message += f"{cols[0]} y {cols[1]}\n"

    message += "\nVariables independientes:\n"
    for cols in independientes:
        message += f"{cols[0]} y {cols[1]}\n"

    return dependientes, independientes, message


def depen_param_media_test(df, dependientes, par_columns):
    from scipy.stats import ttest_rel
    from statsmodels.stats.anova import AnovaRM

    num_grupos = len(set(dependientes).intersection(par_columns))
    dif_muestras = []

    if num_grupos == 1:
        return "No se puede realizar ninguna prueba. Las variables dependientes seleccionadas tienen un solo grupo."
    elif num_grupos == 2:
        # Realizar prueba t de Student pareada
        t_stat, p_value = ttest_rel(
            *[df[column] for column in dependientes if column in par_columns]
        )
        if p_value < 0.05:
            dif_muestras.append(
                f"Variable: {dependientes}, Prueba t de Student pareada, Estadístico t: {t_stat:.4f}, Valor p: {p_value:.4f}"
            )
        return dif_muestras
    elif num_grupos > 2:
        # Realizar ANOVA de medidas repetidas
        anova_results = AnovaRM(df, dependientes, par_columns).fit()
        anova_table = anova_results.summary().tables[0]
        for i in range(1, len(anova_table)):
            variable = anova_table[i][0]
            p_value = float(anova_table[i][4])
            if p_value < 0.05:
                dif_muestras.append(
                    f"Variable: {variable}, ANOVA de medidas repetidas, Valor p: {p_value:.4f}"
                )
        return dif_muestras
    else:
        return "No se encontraron variables dependientes que cumplan con los criterios establecidos."


def indep_param_media_test(df, independientes, par_columns):
    from scipy.stats import ttest_ind
    from statsmodels.stats.anova import AnovaRM

    dif_muestras = []

    if len(par_columns) < 2:
        return "Se requieren al menos dos variables independientes para realizar las pruebas."
    else:
        for i in range(len(par_columns)):
            for j in range(i + 1, len(par_columns)):
                var1 = par_columns[i]
                var2 = par_columns[j]
                grupo1 = df[independientes].loc[df[var1]]
                grupo2 = df[independientes].loc[df[var2]]

                # Realizar prueba t de Student
                t_stat, p_value = ttest_ind(grupo1, grupo2)
                if p_value < 0.05:
                    dif_muestras.append(
                        f"Variables: {var1} y {var2}, Prueba t de Student, Estadístico t: {t_stat:.4f}, Valor p: {p_value:.4f}"
                    )

                # Realizar ANOVA
                anova_results = AnovaRM(
                    df[independientes], [var1, var2], df.index
                ).fit()
                anova_table = anova_results.summary().tables[0]
                p_value = float(anova_table[1][4])
                if p_value < 0.05:
                    dif_muestras.append(
                        f"Variables: {var1} y {var2}, ANOVA, Valor p: {p_value:.4f}"
                    )

    return dif_muestras


def depen_no_param_media_test(df, dependientes, no_par_columns):
    from scipy.stats import wilcoxon, friedmanchisquare

    dif_muestras = []

    if len(dependientes) < 2:
        return "Se requieren al menos dos variables dependientes para realizar la prueba correspondiente."
    else:
        for i in range(len(dependientes)):
            for j in range(i + 1, len(dependientes)):
                stat, p_value = wilcoxon(df[dependientes[i]], df[dependientes[j]])
                if p_value < 0.05:
                    dif_muestras.append(
                        (
                            dependientes[i],
                            dependientes[j],
                            round((1 - p_value) * 100, 2),
                        )
                    )

        if not dif_muestras:
            stat, p_value = friedmanchisquare(*[df[column] for column in dependientes])
            if p_value < 0.05:
                dif_muestras.append(
                    ("Friedman", ", ".join(dependientes), round((1 - p_value) * 100, 2))
                )

    if dif_muestras:
        message = "Variables dependientes con diferencia significativa:\n"
        for cols in dif_muestras:
            message += f"{cols[0]} entre {cols[1]} - Significancia: {cols[2]:.2f}%\n"
        return dif_muestras, message
    else:
        return "No se encontraron diferencias significativas entre las variables dependientes."


def indep_no_param_media_test(df, independientes, no_par_columns):
    from scipy.stats import mannwhitneyu, kruskal

    dif_muestras = []

    if len(independientes) < 2:
        return "Se requieren al menos dos variables independientes para realizar la prueba correspondiente."
    else:
        for i in range(len(independientes)):
            for j in range(i + 1, len(independientes)):
                stat, p_value = mannwhitneyu(
                    df[independientes[i]], df[independientes[j]]
                )
                if p_value < 0.05:
                    dif_muestras.append(
                        (
                            independientes[i],
                            independientes[j],
                            round((1 - p_value) * 100, 2),
                        )
                    )

        if not dif_muestras:
            stat, p_value = kruskal(*[df[column] for column in independientes])
            if p_value < 0.05:
                dif_muestras.append(
                    (
                        "Kruskal-Wallis",
                        ", ".join(independientes),
                        round((1 - p_value) * 100, 2),
                    )
                )

    if dif_muestras:
        message = "Variables independientes con diferencia significativa:\n"
        for cols in dif_muestras:
            message += f"{cols[0]} entre {cols[1]} - Significancia: {cols[2]:.2f}%\n"
        return dif_muestras, message
    else:
        return "No se encontraron diferencias significativas entre las variables independientes."


# Graficas---------------------------------------------------------


def bar_int_plotly(
    bard_df,
    bard_str_column,
    bard_num_col,
    bard_orden,
    size_title_i,
    color_i,
    height_i,
    width_i,
):
    import plotly.graph_objs as go
    import plotly.offline as pyo

    # Crear los datos del gráfico
    data = [
        go.Bar(
            x=bard_df[bard_str_column],
            y=bard_df[bard_df.columns[bard_orden]],
            marker=dict(
                color=bard_df[bard_df.columns[bard_orden]],
                colorscale=color_i,
                colorbar=dict(title=bard_num_col),
            ),
        )
    ]

    # Crear el diseño del gráfico
    layout = go.Layout(
        title=dict(
            text=f"{bard_df.columns[bard_orden].split('_')[0]} {bard_num_col} per {bard_str_column}",
            x=0.5,
            xanchor="center",
            font=dict(size=size_title_i, family="Arial"),
        ),
        xaxis=dict(
            title=bard_str_column,
            tickvals=bard_df[bard_str_column],
            ticktext=bard_df[bard_str_column].astype(int).astype(str),
        ),
        margin=dict(t=100),  # establecer un margen superior de 100
        height=height_i,  # establecer la altura
        width=width_i,  # establecer el ancho
    )

    # Crear la figura y mostrar el gráfico
    fig = go.Figure(data=data, layout=layout)
    return pyo.iplot(fig)

    # def barv_plotly(
    #     barv_df,
    #     barv_str_column,
    #     barv_num_col,
    #     color_v,
    #     ori,
    # ):
    #     import plotly.graph_objs as go
    #     import plotly.offline as pyo

    #     # Crear los datos del gráfico
    #     data = [
    #         go.Bar(
    #             x=barv_df[barv_str_column],
    #             y=barv_df[barv_num_col],
    #             orientation=ori,
    #             marker=dict(
    #                 color=barv_df[barv_num_col],
    #                 colorscale=color_v,
    #                 colorbar=dict(title=barv_num_col),
    #             ),
    #         )
    #     ]

    #     # # Crear el diseño del gráfico
    # layout = go.Layout(
    #     title=dict(
    #         text=f"{barv_df.columns[barv_orden].split('_')[0]} {barv_num_col} per {barv_str_column}",
    #         x=0.5,
    #         xanchor="center",
    #         font=dict(size=size_title_v, family="Arial"),
    #     ),
    #     xaxis=dict(title=barv_str_column),
    #     margin=dict(t=100),  # establecer un margen superior de 100
    #     height=height_v,  # establecer la altura
    #     width=width_v,  # establecer el ancho
    # )

    # Crear la figura y mostrar el gráfico
    fig = go.Figure(data=data)
    return pyo.iplot(fig)


def barv_plotly(barv_df, barv_str_column, barv_num_col, color_v, height_, width_):
    import plotly.graph_objs as go

    # Crear los datos del gráfico
    data = [
        go.Bar(
            x=barv_df[barv_str_column],
            y=barv_df[barv_num_col],
            # orientation=ori,
            marker=dict(
                color=barv_df[barv_num_col],
                colorscale=color_v,
                colorbar=dict(title=barv_num_col),
            ),
        )
    ]

    # Crear el diseño del gráfico
    layout = go.Layout(
        title=dict(
            text=barv_num_col.replace("_", " ").capitalize()
            + " per "
            + barv_str_column.replace("_", " "),
            x=0.5,
            xanchor="center",
            font=dict(size=25, family="Arial"),
        ),
        xaxis=dict(title=barv_str_column),
        yaxis=dict(showticklabels=False),  # Quitar los valores numéricos del eje y
        margin=dict(t=100),  # Establecer un margen superior de 100
        height=height_,  # Ajustar la altura del gráfico
        width=width_,  # Ajustar el ancho del gráfico
    )

    # Crear la figura
    fig = go.Figure(data=data, layout=layout)

    return fig


import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# def mul_bar(df, x_column, y_column, numeric_column):
#     fig = go.Figure()

#     for categoria2 in df[y_column].unique():
#         subset = df[df[y_column] == categoria2]
#         fig.add_trace(
#             go.Bar(x=subset[x_column], y=subset[numeric_column], name=categoria2)
#         )

#     fig.update_layout(barmode="stack")

#     # Mostrar gráfico en Streamlit
#     st.plotly_chart(fig)

# import seaborn as sns


# def mul_bar(df, x_column, y_column, numeric_column):
#     # Crear el gráfico de barras utilizando Seaborn
#     sns.barplot(data=df, x=x_column, y=numeric_column, hue=y_column)

#     # Mostrar gráfico en Streamlit
#     st.pyplot()


import streamlit as st
import plotly.graph_objects as go


def mul_bar(df, x_column, y_column, numeric_column):
    fig = go.Figure()

    for categoria2 in df[y_column].unique():
        subset = df[df[y_column] == categoria2]
        fig.add_trace(
            go.Bar(x=subset[x_column], y=subset[numeric_column], name=categoria2)
        )

    fig.update_layout(barmode="stack")

    # Mostrar gráfico en Streamlit
    return fig


# # Datos de ejemplo
# data = {
#     'Categoría A': ['X', 'Y', 'Z'],
#     'Categoría B': ['X', 'Y', 'Z'],
#     'Variable Numérica': [10, 15, 7]
# }

# # Llamar a la función
# mul_bar(data=data,
#         x_column='Categoría A',
#         y_column='Categoría B',
#         numeric_column='Variable Numérica',
#         title='Relación entre dos variables categóricas y una numérica',
#         xaxis_title='Categorías',
#         yaxis_title='Variable Numérica')


# def mul_bar(
#     bar_m_df,
#     mbar_str_column,
#     mbar_cat_col,
#     mbar_num_col,
#     height_m,
#     width_m,
# ):
#     # Crea la figura
#     fig = go.Figure()

#     # for customer_name in bar_m_df[mbar_str_column].unique():
#     #     customer_df = bar_m_df[bar_m_df[mbar_str_column] == customer_name]
#     fig.add_trace(
#         go.Bar(
#             x=bar_m_df[mbar_str_column],
#             y=bar_m_df[mbar_num_col],
#             name=bar_m_df[mbar_cat_col],
#         )
#     )

#     # Configura el diseño del gráfico
#     fig.update_layout(
#         title={
#             "text": f"{mbar_str_column.split('_')[0]}s with most {bar_m_df.columns[1].replace('_', ' ')} per {mbar_cat_col.replace('_', ' ')}",
#             "font": {"family": "Arial", "size": 32},
#             "x": 0.5,
#             "xanchor": "center",
#         },
#         font=dict(family="Courier New, monospace", size=25, color="#7f7f7f"),
#         yaxis=dict(title=mbar_num_col),
#         xaxis=dict(title=mbar_cat_col),
#         width=width_m,
#         height=height_m,
#     )

#     return fig


import plotly.express as px


def sec_bar_mdf(
    mbar_df, mbar_cat_col, mbar_num_col, mbar_str_column, height_s, width_s
):
    fig = px.bar(
        mbar_df,
        x=mbar_cat_col,
        y=mbar_num_col,
        color=mbar_str_column,
        barmode="stack",
        color_discrete_sequence=px.colors.sequential.Inferno_r,
        title="Ventas por cliente y año",
    )
    fig.update_layout(
        title={
            "text": f"{mbar_str_column.split('_')[0]}s with most {mbar_df.columns[1].replace('_', ' ')} per {mbar_cat_col.replace('_', ' ')}",
            "font": {"family": "Arial", "size": 25},
            "x": 0.5,
            "xanchor": "center",
        },
        font=dict(size=18, family="Arial"),
        width=width_s,
        height=height_s,
        xaxis=dict(
            title=mbar_cat_col,
            tickvals=mbar_df[mbar_cat_col],
            ticktext=mbar_df[mbar_cat_col].astype(int).astype(str),
        ),
        yaxis=dict(title=mbar_num_col.capitalize()),
    )
    return fig


import plotly.express as px


def pie_graph(pie_df, pie_str_column, pie_num_col, color_palette, height_, width_):
    # Obtener los datos para el gráfico
    labels = pie_df[pie_str_column].tolist()
    values = pie_df[pie_num_col].tolist()

    # Crear la figura de tipo "pie" con Plotly Express
    fig = px.pie(pie_df, values=values, names=labels)

    # Obtener la paleta de colores seleccionada
    color_scale = getattr(px.colors.sequential, color_palette)

    # Cambiar la paleta de colores
    fig.update_traces(marker=dict(colors=color_scale))

    # Centrar el título del gráfico
    fig.update_layout(
        title={
            "text": f" {pie_num_col.replace('_', ' ').capitalize()} per {pie_str_column.replace('_', ' ')}",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(family="Arial", size=25),
    )

    # Ajustar el tamaño del texto y la figura
    fig.update_layout(
        title_font_size=28,
        autosize=False,
        width=width_,
        height=height_,
    )

    return fig


def line_graph(line_df, line_x_column, line_y_column, height_, width_):
    import plotly.graph_objs as go

    # Crear los datos del gráfico
    data = [
        go.Scatter(
            x=line_df[line_x_column],
            y=line_df[line_y_column],
            mode="lines",
        )
    ]

    # Crear el diseño del gráfico
    layout = go.Layout(
        title=dict(
            text=line_y_column.replace("_", " ").capitalize()
            + " per "
            + line_x_column.replace("_", " "),
            x=0.5,
            xanchor="center",
            font=dict(size=25, family="Arial"),
        ),
        margin=dict(t=100),  # Establecer un margen superior de 100
        height=height_,  # Ajustar la altura del gráfico
        width=width_,  # Ajustar el ancho del gráfico
    )

    # Verificar si la columna line_x_column tiene un tipo de datos entero (int)
    if line_df[line_x_column].dtype == int or float:
        # Obtener los valores únicos de la columna line_x_column
        x_values = sorted(line_df[line_x_column].unique())
        # Convertir los valores a texto
        x_ticks = [str(value) for value in x_values]
        # Asignar los valores y etiquetas de los ticks al eje x
        layout["xaxis"]["tickvals"] = x_values
        layout["xaxis"]["ticktext"] = x_ticks

    # Crear la figura
    fig = go.Figure(data=data, layout=layout)

    return fig


def line_graph_mult(
    line_df, line_x_column, line_y_column, line_color_column, height_, width_
):
    import plotly.graph_objs as go

    # Crear los datos del gráfico
    data = []
    unique_colors = line_df[line_color_column].unique()
    for color in unique_colors:
        subset_df = line_df[line_df[line_color_column] == color]
        sorted_df = subset_df.sort_values(ascending=False,
            by=line_x_column
        )  # Ordenar por la columna line_x_column
        trace = go.Scatter(
            x=sorted_df[line_x_column],
            y=sorted_df[line_y_column],
            mode="lines",
            name=str(color),
        )
        data.append(trace)

    # Crear el diseño del gráfico
    layout = go.Layout(
        title=dict(
            text=line_y_column.replace("_", " ").capitalize()
            + " per "
            + line_x_column.replace("_", " ")
            + " per "
            + line_color_column,
            x=0.5,
            xanchor="center",
            font=dict(size=25, family="Arial"),
        ),
        xaxis=dict(title=line_x_column),
        margin=dict(t=100),  # Establecer un margen superior de 100
        height=height_,  # Ajustar la altura del gráfico
        width=width_,  # Ajustar el ancho del gráfico
    )

    # Verificar si la columna line_x_column tiene un tipo de datos entero (int)
    if line_df[line_x_column].dtype == int or float:
        # Obtener los valores únicos de la columna line_x_column
        x_values = sorted(line_df[line_x_column].unique())
        # Convertir los valores a texto
        x_ticks = [str(value) for value in x_values]
        # Asignar los valores y etiquetas de los ticks al eje x
        layout["xaxis"]["tickvals"] = x_values
        layout["xaxis"]["ticktext"] = x_ticks

    # Crear la figura
    fig = go.Figure(data=data, layout=layout)

    return fig


# Funcion que devuelve la sumatoria de variables numericas y el registro que mas se repite por consulta
"""# def summary_df(df,main_str_col,string_cols,num_cols):
    
#     main_var=df[main_str_col].unique()
#     df_main_var=pd.DataFrame(main_var)
#     df_main_var.rename({0:main_str_col}, axis=1, inplace=True)
    
#     for n in num_cols:
#         df_main_var['total_year_'+n]=[round((df[df[main_str_col]== i][n].sum()),2) for i in main_var]
#         df_main_var['total_year_'+n]=[round((df[df[main_str_col]== i][n].sum()),2) for i in main_var]
        
#     for s in string_cols:
#         df_main_var['most_frequent_'+s]=[" ".join(str(df[df[main_str_col]== i][s].value_counts().head(1)).split()[:-5]) for i in main_var]
        
#     return df_main_var"""

# Metodo string
".join(str(df[df['customer_name']== 'Toby Braunhardt']['country'].value_counts().head(1)).split()[:-5])"
