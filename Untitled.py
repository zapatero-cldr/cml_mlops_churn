agg_metrics=sys.argv[0]
a=sys.argv[1]
threshold=sys.argv[2]

valores=np.linspace(0,1,6)
import bisect
flecha=bisect.bisect_left(valores, a)

sns.barplot(x=list(range(1,len(agg_metrics)+1)), y="metrics.accuracy", color="grey", data=agg_metrics)
gauge(labels=['CRITICAL','LOW','NOT GOOD','CARE','GOOD'], colors=['#ED1C24','#F18517','#FFCC00','#0063BF','#007A00'], arrow=flecha, title='accuracy ='+str(a)) 
if a < threshold:
  print("se va a reentrenar el modelo, recibirá un correo cuando esté puesto en producción")
else:
  print("el modelo satisface los criterios establecidos")