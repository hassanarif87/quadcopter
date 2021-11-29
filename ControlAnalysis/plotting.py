from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def bode(w, mag, phase):
    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(
        go.Scatter(x=w, y=mag),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=w, y=phase),
        row=2, col=1
    )
    # Update xaxis properties
    fig.update_xaxes(title_text="Freqency")

    # Update yaxis properties
    fig.update_yaxes(title_text="Gain", row=1, col=1)
    fig.update_yaxes(title_text="Phase", row=2, col=1)

    fig.update_layout(height=800, width=1000, title_text="Bode Plot")
    fig.update_xaxes(type="log")
    return fig

def pend_cart_animation(df_results, x0, L = 2):
    x_cart_0 = x0[0]
    y_cart_0 = 0 # constraint cart
    x_pend_0 = x_cart_0  + L *np.cos(x0[2]-np.pi/2)
    y_pend_0 = y_cart_0 + L *np.sin(x0[2]-np.pi/2)
    frames = []
    for index, row in df_results.iterrows():
        x_cart = row['x']
        y_cart = 0
        x_pend = x_cart  + L *np.cos(row['theta']-np.pi/2)
        y_pend = y_cart + L *np.sin(row['theta']-np.pi/2)
        frames.append(go.Frame(data=[go.Scatter(x=[x_cart, x_pend], y=[y_cart, y_pend])]))


    fig = go.Figure(
        data=[go.Scatter(x=[x_cart_0, x_pend_0], y=[y_cart_0, y_pend_0])],
        layout=go.Layout(
            xaxis=dict(range=[0, 5], autorange=False),
            yaxis=dict(range=[0, 5], autorange=False),
            title="Start Title",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None])])]
        )
    )
    fig.update_layout(title='Inverted Pendulum on a Cart',
                    title_x=0.5,
                    #width=600, height=600,
                    xaxis_title='X',
                    yaxis_title='Y',
                    yaxis_range=(-10,10),
                    xaxis_range=(-25,25), #
                    updatemenus=[dict(buttons = [dict(
                                                args = [None, {"frame": {"duration": 5,
                                                                            "redraw": False},
                                                                "fromcurrent": True,
                                                                "transition": {"duration": 0}}],
                                                label = "Play",
                                                method = "animate")],
                                    type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=1.12,
                                    xanchor='right',
                                    yanchor='top')])
    fig.update(frames=frames)

    return fig
