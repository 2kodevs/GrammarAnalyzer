from cmp.functions import analize_grammar

fd = open("Gramar.txt", 'r')
data = fd.read()
fd.close()

values = analize_grammar(data
                         )
fd = open("template.html", 'r', encoding='UTF-8')
data = fd.read()
fd.close()

sec = data.split('%s')
html = []
for i in range(len(sec)):
    html.append(sec[i])
    html.append(values[i])

fd = open('results.html', 'w')
fd.write(''.join(html))
fd.close()



