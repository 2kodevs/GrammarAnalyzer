from cmp.functions import read_grammar, LL1, SLR1Parser
from cmp.pycompiler import Symbol, Epsilon, Sentence, Production, Grammar, Item
from HtmlFormatter import HtmlFormatter as html
from pprint import pprint
from bs4 import BeautifulSoup as beauty


fd = open("Gramar.txt", 'r')
data = fd.read()
fd.close()
G = read_grammar(data)
print(G.startSymbol, G.startSymbol.productions)
# G = make_grammar()

# E, F, T, X, Y = sorted(G.nonTerminals, key=lambda x: x.Name)
# opar, cpar, star, plus, minus, div, num = sorted(G.terminals, key=lambda x: x.Name)

# print(G)

# pprint(is_LL1)
# pprint(M)
# pprint(M[T])
# pprint(type(M[T][opar]))

# fd = open("ll1.html", 'w')
# fd.write(style + html.draw_table(M, 'LL1', G.terminals + [G.EOF], '\t') + '</body>')
# fd.close()



#print(is_SLR1)

values = []

# LL1 + FIRST + FOLLOW + GRAMMAR
is_LL1, M, firsts, follows = LL1(G)

values.append(html.grammar_to_html(G))
values.append(html.firsts_to_html(G, firsts))
values.append(html.follows_to_html(G, follows))
values.append(html.draw_table(M, 'Symbol', G.terminals + [G.EOF], '%s'))
values.append('Grammar is %s LL(1)' % ['not', ''][is_LL1])


# SLR1
parser = SLR1Parser(G)
is_SLR1, action, goto = parser.ok, parser.action, parser.goto
# for i in parser.automaton:
#     if i.idx not in action:
#         action[i.idx] = {}
#     if i.idx not in goto:
#         goto[i.idx] = {}

values.append(html.grammar_to_html(parser.Augmented))

values.append(html.items_collection_to_html(parser.automaton))
values.append(parser.automaton._repr_svg_())
values.append(html.draw_table(action, 'ACTION', parser.Augmented.terminals + [parser.Augmented.EOF], 'I%s'))
values.append(html.draw_table(goto, 'GOTO', parser.Augmented.nonTerminals, 'I%s'))
values.append('Grammar is %s SLR(1)' % ['not', ''][is_SLR1])

values.append('')
# page += html.action_goto_table_to_html(action, parser.G.terminals + [parser.G.EOF], 'ACTIONS')
fd = open("descriptions.html", 'r', encoding='UTF-8')
data = fd.read()
fd.close()

# for i in values:
#     print(i)


# print(data)
sec = data.split('%s')
html = []
for i in range(len(sec)):
    html.append(sec[i])
    html.append(values[i])

fd = open('fff.html', 'w')
fd.write(beauty(''.join(html), 'html.parser').prettify())
fd.close()

