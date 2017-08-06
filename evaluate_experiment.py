import plotly
import plotly.plotly as py
import plotly.graph_objs as go 

plotly.tools.set_credentials_file(username='tisijoe', api_key='kv9QRxjplURljsrq6ppg')

def plotGraph(x, y, titles):
    traces = []
    fig = plotly.tools.make_subplots(rows=1, cols=len(x), subplot_titles=(titles))
    for i in range(len(x)):
        trace = go.Bar(
            x=x[i],
            y=y[i]
        )
        fig.append_trace(trace, 1, i+1)
    
    for i in range(len(x)):
        fig['layout']['yaxis' + str(i+1)].update(title='% Answered Correctly', range=[0,100])  
        fig['layout']['xaxis' + str(i+1)].update(title='Models')  
    
    print fig
    plotly.offline.plot(fig, filename='basic-bar.html')