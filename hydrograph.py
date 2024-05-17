from dash import Dash, dcc, html, dash_table, Input, Output, callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import pandas as pd
import numpy as np
import math
from fractions import Fraction

"""
****************** global variable ******************
"""
#region
utbt = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,
        1.7,1.8,1.9,2] + [2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.5,5]
utbq = [0,0.03,0.1,0.19,0.31,0.47,0.66,0.82,0.93,0.99,1,
        0.99,0.93,0.86,0.78,0.68,0.56,0.46,0.39,0.33,0.28,
        0.207,0.147,0.107,0.077,0.055,0.04,0.029,0.021,
        0.015,0.011,0.005,0]

df = pd.DataFrame(list(zip([0], [0])),columns=["Time(hr)","Flow(cms)"],dtype = float)
horner = pd.DataFrame(list(zip([1565.136,23.715,0.752], [1504.712,21.876,0.685],
                                [1272.369,17.262,0.628], [943.32,9.256,0.549],
                                [739.372,3.391,0.49], [622.442,0,0.442], [612.796,0,0.42] )),
                                columns=["2","5","10","25","50","100","200"],dtype = float,
                                index=["a","b","c"])
horner_ri = horner.reset_index()
horner_ri = horner_ri.rename(columns={"index":"Coef."})

rainfall_total = pd.DataFrame(list(zip([150.6],[207.1],[251.9],[317.9],[374.7],[438.4],[510.5], )),
                                columns=["2","5","10","25","50","100","200"],dtype = float)
phi_table = pd.DataFrame(list(zip([1.69],[1.78],[1.82],[1.87],[1.90],[1.92],[1.94], )),
                                columns=["2","5","10","25","50","100","200"],dtype = float)

# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

#endregion


"""
****************** unit Hydrograph ******************
"""
#region
header1 = html.H4(
    "Unit Hydrograph", className="bg-primary text-white p-2 mb-2 text-center"
)

table_unit = html.Div(
    dash_table.DataTable(
        id="table_unit",
        columns=[{"name": i, "id": i, "deletable": False, "type": "numeric"} for i in df.columns],
        data=df.to_dict("records"),
        page_size=30,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)

dropdown = html.Div(
    dbc.Select(
        ["Dimensionless", "Triangular"],
        "Dimensionless",
        id="unit-method-select",
    ),
    className="py-2",
)

number_input_tc = dbc.FormFloating(
    [
        dbc.Input(id="input_tc", type="number", min=0, value=3.89),
        dbc.Label("Time of concentration(hr)"),
    ]
)
number_input_a = dbc.FormFloating(
    [
        dbc.Input(id="input_a", type="number", min=0, value=52.51),
        dbc.Label("Subcatchment Area(km2)"),
    ]
)
number_input_ut = dbc.FormFloating(
    [
        dbc.Input(id="input_ut", type="number", min=0.01, max=5, value=0.8),
        dbc.Label("Unit Duration of Rainfall D (hr)"),
    ]
)
number_input_dt_u = dbc.FormFloating(
    [
        dbc.Input(id="input_dt_u", type="number", min=0, max=5, value=0.4),
        dbc.Label("Delta T (hr)"),
    ]
)
number_output_tp = dbc.FormFloating(
    
    [
        dbc.Input(id="output_tp",type="number", disabled=True),
        dbc.Label("time of peak flow(hr)"),
    ]
)
number_output_tb = dbc.FormFloating(
    [
        dbc.Input(id="output_tb",type="number", readonly=True),
        dbc.Label("time of basic flow(hr)"),
    ]
)
number_output_qp = dbc.FormFloating(
    [
        dbc.Input(id="output_qp",type="number", readonly=True),
        dbc.Label("Peak Flow(cms)"),
    ]
)
button = html.Div(
    [
        dbc.Button(
            "Calculate", id="example-button", className="me-2", n_clicks=0,
            outline=True, color="primary"
        ),
        html.Span(id="example-output", style={"verticalAlign": "middle"}),
    ]
)
inputs_u = dbc.Card(
    [
        dbc.CardHeader("Input"),
        dbc.CardBody(
            [number_input_tc, number_input_a, number_input_ut, number_input_dt_u, dropdown]
        ),
    ],
    # style={"width": "18rem"},
)
outputs_u = dbc.Card(
    [
        dbc.CardHeader("Output"),
        dbc.CardBody(
            [number_output_tp, number_output_tb, number_output_qp]
        ),
    ],
    # style={"width": "18rem"},
)

tab1 = dbc.Tab([dcc.Graph(id="line-chart-unit")], label="Line Chart", tab_id="Tab_Unit_Chart")
tab2 = dbc.Tab([table_unit], label="Table", className="p-4", tab_id="Tab_Unit_Table")
tabs = dbc.Card(dbc.Tabs([tab1, tab2], id="Tabs_Unit"))
#endregion


"""
****************** Rainfall Hyetograph ******************
"""
#region
header2 = html.H4(
    "Rainfall Hyetograph", className="bg-primary text-white p-2 mb-2 text-center"
)
number_input_cn = dbc.FormFloating(
    [
        dbc.Input(id="input_cn", type="number", min=0, value=92, max=100),
        dbc.Label("Curve Number"),
    ]
)
number_input_dt_r = dbc.FormFloating(
    [
        dbc.Input(id="input_dt_r", type="number", min=0.1, value=0.8),
        dbc.Label("Delta T(hr)"),
    ]
)
number_input_rd = dbc.FormFloating(
    [
        dbc.Input(id="input_rd", type="number", min=0, value=24),
        dbc.Label("Rainfall duration(hr)"),
    ]
)
number_input_phi = dbc.FormFloating(
    [
        dbc.Input(id="input_phi", type="number", min=0, value=2),
        dbc.Label("Phi(mm/hr)"),
    ]
)
dropdown_r = html.Div(
    dbc.Select(
        ["Accumulation", "Phi"],
        "Accumulation",
        id="loss-method-select",
    ),
    className="py-2",
)
number_output_y = dbc.FormFloating(

    [
        dbc.Input(id="output_y", type="number", disabled=True),
        dbc.Label("Y(mm)"),
    ]
)
number_output_ia = dbc.FormFloating(

    [
        dbc.Input(id="output_ia", type="number", disabled=True),
        dbc.Label("Ia(mm)"),
    ]
)
table_horner = html.Div(
    dash_table.DataTable(
        id="table_horner",
        columns=[{"name": i, "id": i, "deletable": False, "type": "numeric"} for i in horner_ri.columns],
        data=horner_ri.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table_rainfall_total = html.Div(
    dash_table.DataTable(
        id="table_rainfall_total",
        columns=[{"name": i, "id": i, "deletable": False, "type": "numeric"} for i in rainfall_total.columns],
        data=rainfall_total.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table_phi = html.Div(
    dash_table.DataTable(
        id="table_phi",
        columns=[{"name": i, "id": i, "deletable": False, "type": "numeric"} for i in phi_table.columns],
        data=phi_table.to_dict("records"),
        page_size=10,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table_rain_ratio = html.Div(
    dash_table.DataTable(
        id="table_rain_ratio",
        columns=[{"name": "Time(hr)", "id": "0", "deletable": False}]
                +[{"name": i, "id": i, "deletable": False} for i in horner.columns],
        data=df.to_dict("records"),
        page_size=1440,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table_rain_P = html.Div(
    dash_table.DataTable(
        id="table_rain_P",
        columns=[{"name": "Time(hr)", "id": "0", "deletable": False}]
                +[{"name": i, "id": i, "deletable": False} for i in horner.columns],
        data=df.to_dict("records"),
        page_size=1440,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table_rain_Pe = html.Div(
    dash_table.DataTable(
        id="table_rain_Pe",
        columns=[{"name": "Time(hr)", "id": "0", "deletable": False}]
                +[{"name": i, "id": i, "deletable": False} for i in horner.columns],
        data=df.to_dict("records"),
        page_size=1440,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)

inputs_r = dbc.Card(
    [
        dbc.CardHeader("Input"),
        dbc.CardBody(
            [number_input_cn, number_input_dt_r, number_input_rd, dropdown_r]
        ),
    ],
    # style={"width": "18rem"},
)
outputs_r = dbc.Card(
    [
        dbc.CardHeader("Output"),
        dbc.CardBody(
            [number_output_y, number_output_ia]
        ),
    ],
    # style={"width": "18rem"},
)

tab1_r = dbc.Tab([table_horner], label="Horner", className="p-4",label_style={"color": "#00AEF9"})
tab1_r1 = dbc.Tab([table_rainfall_total], label="Rainfall", className="p-4",label_style={"color": "#00AEF9"})
tab1_r2 = dbc.Tab([table_phi], label="Avg Infiltration(mm)", className="p-4",label_style={"color": "#00AEF9"})
tab2_r = dbc.Tab([table_rain_ratio], label="Series(%)", className="p-4")
tab3_r = dbc.Tab([table_rain_P], label="Series(P)", className="p-4")
tab4_r = dbc.Tab([table_rain_Pe], label="Series(Pe)", className="p-4")
tab5_r = dbc.Tab(dbc.Row([dbc.Col(dcc.Graph(id="Distrubution"))]), label="Chart(%)")
tab6_r = dbc.Tab([dcc.Graph(id="Precipitation")], label="Chart(P)")
tab7_r = dbc.Tab([dcc.Graph(id="Precipitation(loss)")], label="Chart(Pe)")
tabs_r = dbc.Card(dbc.Tabs([tab1_r,tab1_r1, tab1_r2, tab2_r, tab3_r, tab4_r, tab5_r, tab6_r, tab7_r]))

#endregion

"""
****************** Flow Hydrograph ******************
"""
#region
header3 = html.H4(
    "Flow Hydrograph", className="bg-primary text-white p-2 mb-2 text-center"
)
select_yr_Q = html.Div(
    dcc.Dropdown(
        ["2","5","10","25","50","100","200"],
        ["10"],
        id="select_yr_Q",
        placeholder="Select Multiple Return Period (yr)",
        multi=True
    )
)
number_output_Qp = dbc.FormFloating(
    [
        dbc.Input(id="output_Qp", type="number", disabled=True),
        dbc.Label("Qp(cms)"),
    ]
)
number_output_Tp = dbc.FormFloating(
    [
        dbc.Input(id="output_Tp", type="number", disabled=True),
        dbc.Label("Tp(hr)"),
    ]
)
table_flow = html.Div(
    dash_table.DataTable(
        id="table_flow",
        columns=[{"name": "Time(hr)", "id": "0", "deletable": False}]
                +[{"name": i, "id": i, "deletable": False} for i in horner.columns],
        data=df.to_dict("records"),
        page_size=1440,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
table_unit_S = html.Div(
    dash_table.DataTable(
        id="table_unit_S",
        columns=[{"name": i, "id": i, "deletable": False, "type": "numeric"} for i in df.columns],
        data=df.to_dict("records"),
        page_size=100,
        editable=True,
        cell_selectable=True,
        sort_action="native",
        style_table={"overflowX": "auto"},
    ),
    className="dbc-row-selectable",
)
table_Qp = html.Div(
    dash_table.DataTable(
        id="table_Qp",
        columns=[{"name": "Flow(cms)", "id": "0", "deletable": False}]
                +[{"name": i, "id": i, "deletable": False} for i in horner.columns],
        data=df.to_dict("records"),
        page_size=2000,
        editable=True,
        cell_selectable=True,
        # filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        # row_selectable="multi",
    ),
    className="dbc-row-selectable",
)
outputs_f = dbc.Card(
    [
        dbc.CardHeader("Output"),
        dbc.CardBody(
            #[number_output_Qp, number_output_Tp]
            []
        ),
    ],
    # style={"width": "18rem"},
)
tab_f1 = dbc.Tab([dcc.Graph(id="Flow")], label="Chart")
tab_f2 = dbc.Tab([table_flow], label="Table(Q)", className="p-4")
tab_f3 = dbc.Tab([table_Qp], label="Table(Qp)", className="p-4")
tab_f4 = dbc.Tab([table_unit_S], label="Table(unit_S_curve_method)", className="p-4")
tab_f5 = dbc.Tab([dcc.Graph(id="line-chart-unitS")], label="Chart(unit_S_curve_method)", tab_id="Tab_Unit_ChartS")
tabs_f = dbc.Card(dbc.Tabs([tab_f1, tab_f2, tab_f3, tab_f4, tab_f5], id="Tabs_flow"))
#endregion

"""
******************** LAYOUT **********************
"""
#region
app.layout = dbc.Container(
    [
        header1,
        dbc.Row(
            [
                dbc.Col(
                    [
                        inputs_u,
                        outputs_u,
                    ],
                    width=4,
                ),
                dbc.Col([tabs], width=8),
            ]
        ),

        header2,
        dbc.Row(
            [
                dbc.Col(
                    [
                        inputs_r,
                        outputs_r,
                    ],
                    width=4,
                ),
                dbc.Col([tabs_r], width=8),
            ]
        ),

        header3,
        dbc.Row(
            [
                dbc.Col(
                    [
                        outputs_f,
                    ],
                    width=4,
                ),
                dbc.Col([tabs_f], width=8),
            ]
        ),

        ThemeChangerAIO(aio_id="theme")
    ],
    fluid=True,
    className="dbc",
)
#endregion


"""
******************** Function **********************
"""
#region
#unit
@app.callback(
    [
        Output("output_tp", "value"),
        Output("output_tb", "value"),
        Output("output_qp", "value"),
        Output("table_unit", "data"),
        Output("line-chart-unit", "figure"),

    ],
    [
        Input("input_tc", "value"),
        Input("input_a", "value"),
        Input("input_ut", "value"),
        Input("input_dt_u", "value"),
        Input("table_unit", "data"),
        Input("unit-method-select","value"),
        Input(ThemeChangerAIO.ids.radio("theme"), "value"),
    ]
)
def cal_unit(tc, a, ut, dt, data, method, theme):
    output = []
    
    if (tc in [None,0]) or (a in [None,0]) or (dt in [None,0]) or (ut in [None,0]):
        output = ["", "", "", data, {}]
    else:
        dt = get_fraction(dt)
        tp = round(ut*0.5+tc*0.6,4)
        tb = round(2.67*tp,4)
        qp = round((0.208*a)/tp,4)

        ux, uy = cal_uxuy(tc,a,ut,dt,method)

        if method == "Triangular":
            data_fig = [{'Time(hr)': 0, 'Flow(cms)': 0},
                {'Time(hr)': tp, 'Flow(cms)': qp},
                {'Time(hr)': tb, 'Flow(cms)': 0},]
            #ux = [x for x in np.arange(0, tb, dt)]
            #ux.append(dt*math.ceil(tb/dt))
            #uy = np.interp(ux, [0,tp,tb], [0,qp,0])
            #ux = [round(x,4) for x in ux]
            #uy = [round(x,4) for x in uy]
            dfu = pd.DataFrame(list(zip(ux, uy)),columns=["Time(hr)","Flow(cms)"],dtype = float)
            data_new = dfu.to_dict("records")

        else:
            #ux = [x for x in np.arange(0, tp*5, dt)]
            #ux.append(dt*math.ceil((tp*5)/dt))
            #ux = [x/tp for x in ux]
            #uy = np.interp(ux, utbt, utbq)
            #ux = [round(x*tp,4) for x in ux]
            #uy = [round(x*qp,4) for x in uy]
            
            dfu = pd.DataFrame(list(zip(ux, uy)),columns=["Time(hr)","Flow(cms)"],dtype = float)
            data_new = dfu.to_dict("records")
            data_fig = data_new

        dff = pd.DataFrame.from_records(data_fig)
        fig = px.line(
            dff,
            x="Time(hr)",
            y="Flow(cms)",
            markers=True,
            # color="continent",
            # line_group="country",
            line_shape="linear" if method == "Triangular" else "spline",
            template=template_from_url(theme),
        )
        output = [tp, tb, qp, data_new, fig]
    return output

#rain
@app.callback(
    [
        Output("output_y", "value"),
        Output("output_ia", "value"),
        Output("table_rain_ratio", "data"),
        Output("table_rain_P", "data"),
        Output("table_rain_Pe", "data"),
        Output("Distrubution","figure"),
        Output("Precipitation","figure"),
        Output("Precipitation(loss)","figure")
    ],
    [
        Input("input_cn", "value"),
        Input("input_rd", "value"),
        Input("input_dt_r", "value"),
        Input("table_rainfall_total", "data"),
        Input("table_phi", "data"),
        Input("table_rain_ratio", "data"),
        Input("table_rain_P", "data"),
        Input("table_rain_Pe", "data"),
        Input("loss-method-select","value"),
        Input(ThemeChangerAIO.ids.radio("theme"), "value"),
    ]
)
def cal_rainfall(cn, rd, dt_r, data_rain, data_phi, data_r, data_p, data_pe, method, theme):
    output = []
    if (cn in [None,0]) or (dt_r in [None,0]) or (data_r in [None,0]):
        output = ["", "", [], [], [], {}, {}, {}]
    else:
        #dt_r = get_fraction(dt_r)
        y = round(25.4*((1000/cn)-10),4)
        ia = round(y*0.2,4)
        t1x = [x for x in np.arange(dt_r, rd+dt_r, dt_r)]
        t1x = [round(x,4) for x in t1x]
        t1_P=[]
        t1_Pe=[]
        t1_R=[]
    
        for i in ["2","5","10","25","50","100","200"]:
            t1_intensity = [cal_horner(get_horner_coef(i),x*60) for x in t1x]
            t1_intensity = [round(x,4) for x in t1_intensity]

            t1_depth = [cal_horner(get_horner_coef(i),x*60)*x*60 for x in t1x]
            t1_dp = [t1_depth[0]]
            for j in range(1,len(t1_depth)):
                t1_dp.append(t1_depth[j]-t1_depth[j-1])

            t1_alter = alternate_blocks(t1_dp)
            
            t1_ratio = [x/sum(t1_alter) for x in t1_alter]
            t1_Pre = [x*data_rain[0][i] for x in t1_ratio]
            t1_ratio = [round(x,4) for x in t1_ratio]
            
            t1_Pre = [round(x,4) for x in t1_Pre]
            if method == "Accumulation":
                t1_Pe_acc = []
                t1_Rain = []
                for j in range(len(t1_Pre)+1):
                    temp_p = sum(t1_Pre[:j])
                    temp_Ia = temp_p if temp_p<ia else ia
                    temp_F = y*(temp_p-temp_Ia)/(temp_p-ia+y) if temp_p>ia else 0
                    temp_pe = temp_p-temp_Ia-temp_F
                    t1_Pe_acc.append(temp_pe)
                    #print(temp_p, temp_Ia, temp_F, temp_pe)
                    if j != 0:
                        temp_rain = t1_Pe_acc[j] - t1_Pe_acc[j-1]
                        t1_Rain.append(temp_rain)
                    
                t1_Rain = [round(x,4) for x in t1_Rain]
            else:
                t1_Rain = [round(x-(data_phi[0][i]*dt_r),4) if (x-(data_phi[0][i]*dt_r))>0 else 0 for x in t1_Pre]
                
            t1_R.append(t1_ratio)
            t1_P.append(t1_Pre)
            t1_Pe.append(t1_Rain)

        dft_R = pd.DataFrame(list(zip(t1x,t1_R[0],t1_R[1],t1_R[2],t1_R[3],t1_R[4],t1_R[5],t1_R[6])),columns=["0","2","5","10","25","50","100","200"],dtype = float)
        data_R = dft_R.to_dict("records")
        dft_P = pd.DataFrame(list(zip(t1x,t1_P[0],t1_P[1],t1_P[2],t1_P[3],t1_P[4],t1_P[5],t1_P[6])),columns=["0","2","5","10","25","50","100","200"],dtype = float)
        data_P = dft_P.to_dict("records")
        dft_Pe = pd.DataFrame(list(zip(t1x,t1_Pe[0],t1_Pe[1],t1_Pe[2],t1_Pe[3],t1_Pe[4],t1_Pe[5],t1_Pe[6])),columns=["0","2","5","10","25","50","100","200"],dtype = float)
        data_Pe = dft_Pe.to_dict("records")

        fig_R = px.bar(
            dft_R,
            x="0",
            y=["2","5","10","25","50","100","200"],
            barmode="overlay",
            labels={"0":"Time(hr)"},
            template=template_from_url(theme),
        )
        fig_R.update_layout(yaxis=dict(title="Percentage(%)",tickformat=".2%"),
                            xaxis=dict(tickmode='linear',dtick=dt_r),
                            legend_title_text="Return Period(yr)",
                            autosize=True)

        fig_P = px.bar(
            dft_P,
            x="0",
            y=["2","5","10","25","50","100","200"],
            barmode="overlay",
            labels={"0":"Time(hr)"},
            template=template_from_url(theme),
        )
        fig_P.update_layout(yaxis=dict(title="Precipitation(mm)"),
                            xaxis=dict(tickmode='linear',dtick=dt_r),
                            legend_title_text="Return Period(yr)",
                            autosize=True)

        fig_Pe = px.bar(
            dft_Pe,
            x="0",
            y=["2","5","10","25","50","100","200"],
            barmode="overlay",
            labels={"0":"Time(hr)"},
            template=template_from_url(theme),
        )
        fig_Pe.update_layout(yaxis=dict(title="Effective Precipitation(mm)"),
                            xaxis=dict(tickmode='linear',dtick=dt_r),
                            legend_title_text="Return Period(yr)",
                            autosize=True)
        output = [y, ia, data_R, data_P, data_Pe, fig_R, fig_P, fig_Pe]

    return output

#flow
@app.callback(
    [
        #Output("output_Qp", "value"),
        #Output("output_Tp", "value"),
        Output("table_flow", "data"),
        Output("table_Qp", "data"),
        Output("Flow","figure"),
        Output("table_unit_S", "data"),
        Output("line-chart-unitS", "figure"),
    ],
    [
        Input("input_tc", "value"),
        Input("input_a", "value"),
        Input("input_ut", "value"),
        Input("input_dt_u", "value"),
        Input("unit-method-select","value"),
        Input("input_dt_r", "value"),
        Input("table_rain_Pe", "data"),
        Input("table_unit", "data"),
        Input(ThemeChangerAIO.ids.radio("theme"), "value"),
    ]
)
def cal_flow(tc, a, ut, dt, method, dt_r, data_pe, data_u, theme):
    fig={}
    fig_us={}
    data_qq=[]
    data_qp=[]
    data_us=[]
    qqbig = []
    qqmax = []
    if (dt in [None,0]) or (dt_r in [None,0]):
        return [ [], [], {}, [], {}]
    #dt_r = get_fraction(dt_r)
    #dt = get_fraction(dt)
    if dt == dt_r:
        
        #step1
        t1x = [x for x in np.arange(0, dt_r*(len(data_pe)+len(data_u)-1), dt_r)]
        t1x = [round(x,4) for x in t1x]
        #step2
        for i in ["2","5","10","25","50","100","200"]:
            qq = [0 for _ in range(len(data_pe)+len(data_u)-1)]
            ii=0
            for Pe in data_pe:
                for k in range(ii,ii+len(data_u)):
                    qq[k] = qq[k] + Pe[i]*data_u[k-ii]['Flow(cms)']
                ii+=1
            qq = [round(x,4) for x in qq]+[0]
            qqbig.append(qq)
            qqmax.append([max(qq)])
    else:
        #step1
        gcd = gcd_with_fraction2(float(dt),float(dt_r))
        off0 = int(dt/gcd)
        off1 = int(dt_r/gcd)
        #print(off0, off1)
        ux0, uy0 = cal_uxuy(tc,a,ut,dt,method)

        ux = [x for x in np.arange(0, ux0[-1]+gcd, gcd)]
        uy = np.interp(ux, ux0, uy0)

        sy = [0 for _ in range(len(uy)*100)]
        ii=0
        for i in range(10000):
            for j in range(ii,ii+len(uy)):
                #print(j,ii,j-ii)
                sy[j] = sy[j] + uy[j-ii]
                
            
            ii+=off0
            if sy.count(max(sy)) >= 4:
                print(f'S curve Method loop {i} times')
                break
                
        sy1 = [0 for _ in range(off1)] + sy
        sy2 = sy[:]
        for i in range(len(sy)):
            sy2[i] = sy2[i]-sy1[i]
        sy3 = [round((dt*x)/dt_r,4) if x>0 else 0 for x in sy2 ]
        ls = [i for i,e in enumerate(sy3) if e !=0]
        sy3 = sy3[:ls[-1]+1]
        sx3 = [x for x in np.arange(0, gcd*len(sy3), gcd)]
        sx4 = [x for x in np.arange(0, gcd*len(sy3)+dt_r, dt_r)]
        sy4 = np.interp(sx4, sx3, sy3)
        sx4 = [round(x,4) for x in sx4]
        sy4 = [round(x,4) for x in sy4]

        #step2
        t1x = [x for x in np.arange(0, dt_r*(len(data_pe)+len(sx4)-1), dt_r)]
        t1x = [round(x,4) for x in t1x]
        #print(sy4)

        #step3
        for i in ["2","5","10","25","50","100","200"]:
            qq = [0 for _ in range(len(data_pe)+len(sx4)-1)]
            ii=0
            for Pe in data_pe:
                for k in range(ii,ii+len(sx4)):
                    qq[k] = qq[k] + Pe[i]*sy4[k-ii]
                ii+=1
            qq = [round(x,4) for x in qq]+[0]
            qqbig.append(qq)
            qqmax.append([max(qq)])

        #step4
        dfus = pd.DataFrame(list(zip(sx4, sy4)),columns=["Time(hr)","Flow(cms)"],dtype = float)
        data_us = dfus.to_dict("records")
        fig_us = px.line(
            data_us,
            x="Time(hr)",
            y="Flow(cms)",
            markers=True,
            # color="continent",
            # line_group="country",
            line_shape="linear" if method == "Triangular" else "spline",
            template=template_from_url(theme),
        )
    
    # u x r >>> Flow
    #print(qqbig)
    dft_qq = pd.DataFrame(list(zip(t1x,qqbig[0],qqbig[1],qqbig[2],qqbig[3],qqbig[4],qqbig[5],qqbig[6])),columns=["0","2","5","10","25","50","100","200"],dtype = float)
    data_qq = dft_qq.to_dict("records")
    dft_qp = pd.DataFrame(list(zip(t1x,qqmax[0],qqmax[1],qqmax[2],qqmax[3],qqmax[4],qqmax[5],qqmax[6])),columns=["0","2","5","10","25","50","100","200"],dtype = float)
    data_qp = dft_qp.to_dict("records")
    fig = px.line(
        dft_qq,
        x="0",
        y=["2","5","10","25","50","100","200"],
        markers=False,
        template=template_from_url(theme),
        labels={"0":"Time(hr)"},
        line_shape="spline" ,
    )
    fig.update_layout(yaxis=dict(title="Flow(cms)"),
                        xaxis=dict(tickmode='linear',dtick=max(dt_r,0.4)),
                        legend_title_text="Return Period(yr)",
                        autosize=True)
    



    output = [ data_qq, data_qp, fig, data_us, fig_us]

    return output

# other
def cal_uxuy(tc, a, ut, dt,method):
    tp = round(ut*0.5+tc*0.6,4)
    tb = round(2.67*tp,4)
    qp = round((0.208*a)/tp,4)
    if dt==0:
        return [], []
    if method == "Triangular":        
        ux = [x for x in np.arange(0, tb, dt)]
        ux.append(dt*math.ceil(tb/dt))
        uy = np.interp(ux, [0,tp,tb], [0,qp,0])
        ux = [round(x,4) for x in ux]
        uy = [round(x,4) for x in uy]
    else:
        ux = [x for x in np.arange(0, tp*5, dt)]
        ux.append(dt*math.ceil((tp*5)/dt))
        ux = [x/tp for x in ux]
        uy = np.interp(ux, utbt, utbq)
        ux = [round(x*tp,4) for x in ux]
        uy = [round(x*qp,4) for x in uy]
    return ux, uy


def cal_horner(coef,t):
    return coef[0]/(t+coef[1])**coef[2]

def get_horner_coef(yr):
    return [horner[yr]["a"],horner[yr]["b"],horner[yr]["c"]]

def alternate_blocks(lst):
    sorted_lst = sorted(lst, reverse=True)
    result = [sorted_lst.pop(0)]
    while sorted_lst:
        result.append(sorted_lst.pop(0))
        if sorted_lst:
            result.insert(0, sorted_lst.pop(0))
    return result

def get_fraction(decimal_input):
    return Fraction(decimal_input).limit_denominator()

def gcd_with_fraction2(decimal_input1, decimal_input2):
    fraction1 = get_fraction(decimal_input1)
    fraction2 = get_fraction(decimal_input2)
    
    common_denominator = math.lcm(fraction1.denominator, fraction2.denominator)
    numerator1 = fraction1.numerator * (common_denominator // fraction1.denominator)
    numerator2 = fraction2.numerator * (common_denominator // fraction2.denominator)
    
    return Fraction(math.gcd(numerator1, numerator2), common_denominator)
#endregion


if __name__ == "__main__":
    app.run_server(debug=True)
