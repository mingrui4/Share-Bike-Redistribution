import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# network analysis
path_file='201709_top100.csv'
predict = pd.read_csv(path_file, index_col = 'Time')
path_file='201709_station.csv'
station = pd.read_csv(path_file)
G = nx.DiGraph()
G_flow = nx.DiGraph()
time = 17
id_list = []
sum = 0
pos_dic ={}
lab_dic = {}
size_list = []
color_list = []
center_x = 0
center_y = 0
for i in range(20):
    temp_id = predict.iloc[:, i].name
    print(temp_id)
    temp_station = station[(station['start station id'] == int(temp_id))]
    temp_lan = temp_station['start station latitude'].iloc[0]
    temp_long = temp_station['start station longitude'].iloc[0]
    temp_name = temp_station['start station name'].iloc[0]
    spam = int(predict.iloc[time, i]*100)
    G.add_node(temp_id, x=temp_lan, y=temp_long, spam=spam)
    G_flow.add_node(temp_id, x=temp_lan, y=temp_long, spam=spam)
    id_list.append(temp_id)
    sum += spam
    pos_dic[temp_id] = [temp_lan, temp_long]
    lab_dic[temp_id] = temp_name
    size_list.append(abs(spam))
    if predict.iloc[time, i] >=0:
        color = 'r'
    else:
        color = 'b'
    color_list.append(color)
    color_list.append(color)
    center_x += temp_lan
    center_y += temp_long
center_lan = center_x/20
center_long = center_y/20
print(center_lan)
print(center_long)
G.add_node(10000, x=center_lan, y=center_long, spam=0 - sum)
G_flow.add_node(10000, x=center_lan, y=center_long, spam=0 - sum)
id_list.append(10000)
pos_dic[10000] = [center_lan, center_long]
lab_dic[temp_id] = 'center'
size_list.append(0 - sum)
color_list.append('g')
# cost = [([0] * 24) for i in range(100)]
# for i in len(id_list):
#     for j in len(id_list):
#         cost[i][j] = np.sqrt(np.square(G.node[id_list[i]]['x'] - G.node[id_list[j]]['x']) + np.square(G.node[id_list[i]]['y'] - G.node[id_list[j]]['y']))
for i in range(len(id_list)):
    last = id_list.pop()
    if len(id_list) == 0:
        break
    for j in range(len(id_list)):
        a = np.sqrt(np.square(G.node[last]['x'] - G.node[id_list[j]]['x']) + np.square(G.node[last]['y'] - G.node[id_list[j]]['y']))
        b = int(a * 1000)
        G.add_edge(last, id_list[j], cost=b)
        G.add_edge(id_list[j], last, cost=b)

nx.draw_networkx_labels(G, pos = pos_dic, labels=lab_dic, font_size=15)
nx.draw_networkx_nodes(G, pos=pos_dic, node_size= size_list,node_color=color_list)
nx.draw_networkx_edges(G, pos=pos_dic, width=0.1)
# nx.draw(G,with_labels=True,pos=pos_dic)
plt.show()
flowCost, flowDict = nx.network_simplex(G, demand='spam',weight='cost')
all_df = False
flow_label = {}
for i in flowDict:
    for j in flowDict[i]:
        if flowDict[i][j] == 0:
            continue
        try:
            start_station = station[(station['start station id'] == int(i))]
            start_lan = start_station['start station latitude'].iloc[0]
            start_long = start_station['start station longitude'].iloc[0]
            start_name = start_station['start station name'].iloc[0]
        except:
            start_lan = center_lan
            start_long = center_long
            start_name = 'center'
        try:
            stop_station = station[(station['start station id'] == int(j))]
            stop_lan = stop_station['start station latitude'].iloc[0]
            stop_long = stop_station['start station longitude'].iloc[0]
            stop_name = stop_station['start station name'].iloc[0]
        except:
            stop_lan = center_lan
            stop_long = center_long
            stop_name = 'center'

        data = np.array([i,start_lan,start_long,start_name,j,stop_lan,stop_long,stop_name,flowDict[i][j]])
        s = pd.Series(data, index=['start', 'start_lan','start_long','star_name','end','end_lan','end_long','end_name','weight'])
        G_flow.add_edge(i, j,weight=flowDict[i][j])
        flow_label[i,j] = flowDict[i][j]/100
        if all_df is False:
            all_df = s
        else:
            all_df = pd.concat([all_df, s], join='outer', axis=1)
min_flow = all_df.transpose()
print(min_flow)
min_flow.to_csv('minflow.csv',index = True)
nx.draw_networkx_labels(G_flow, pos = pos_dic, labels=lab_dic, font_size=15)
nx.draw_networkx_nodes(G_flow, pos=pos_dic, node_size= size_list,node_color=color_list)
nx.draw_networkx_edges(G_flow, pos=pos_dic, width=1)
nx.draw_networkx_edge_labels(G_flow, pos=pos_dic, edge_labels=flow_label)
# nx.draw(G,with_labels=True,pos=pos_dic)
plt.show()
    # print(temp_lan)
    # print(temp_long)