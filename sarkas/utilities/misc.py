"""Miscellaneous routines."""

from pandas import concat, Series


def add_col_to_df(df, data, column_name):
    """Routine to add a column of data to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to which data has to be added.

    data: numpy.ndarray
        Data to be added.

    column_name: str
        Name of the column to be added.

    Returns
    -------
    _ : pandas.DataFrame
        Original `df` concatenated with the `data`.

    Note
    ----
        It creates a `pandas.Series` from `data` using `df.index`. Then it uses `concat` to add the column.

    """
    col_data = Series(data, index=df.index)

    return concat([df, col_data.rename(column_name)], axis=1)
