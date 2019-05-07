from cmp.pycompiler import Symbol, Epsilon, Sentence, Production, Grammar, Item
from cmp.functions import ShiftReduceParser
from cmp.automata import State

class HtmlFormatter:
    @staticmethod
    def symbol_to_html(s):
        return f'<span class="grammarSymbol">{s}</span>'

    @staticmethod
    def epsilon_to_html(e):
        return f'<span class="grammarEpsilon">{e}</span>'

    @staticmethod
    def sentence_to_html(s):
        return HtmlFormatter.collection_to_html(s, ' ')

    arrow_to_html = '<span class="grammarArrow"> → </span>'

    @staticmethod
    def production_to_html(p):
        return ('%s%s%s' % (HtmlFormatter.symbol_to_html(p.Left), HtmlFormatter.arrow_to_html,
                            HtmlFormatter.custom_to_html(p.Right)))

    or_to_html = '<span class="grammarOr"> | </span>'

    @staticmethod
    def nt_productions_to_html(nt):
        return '%s%s%s' % (HtmlFormatter.symbol_to_html(nt), HtmlFormatter.arrow_to_html,
                            HtmlFormatter.collection_to_html([p.Right for p in nt.productions], HtmlFormatter.or_to_html))

    @staticmethod
    def action_to_html(a):
        action, tag = a
        return '%s%s' % ('<span style="color: brown">S</span>' if action == ShiftReduceParser.SHIFT else '<span style="color: green">OK</span>' if action == ShiftReduceParser.OK else '',
                        HtmlFormatter.custom_to_html(tag))

    dot_to_html = '<span class="grammarDot"> • </span>'

    @staticmethod
    def draw_rows(coll, headers, symbol):
        s1 = f'<tr>%s</tr>'
        s2 = f'<tr class="alt">%s</tr>\n'
        rh = f'<td>{symbol}</td>'
        rows = []
        for s in coll:
            column = [rh % HtmlFormatter.custom_to_html(s)]
            for h in headers:
                c = HtmlFormatter.draw_cell(coll[s], h)
                column.append(c % ('---' if h not in coll[s] else '<p>%s</p>' % HtmlFormatter.collection_to_html(coll[s][h], '<p></p>')))
            rows.append(s1 % ''.join(column))
            s1, s2 = s2, s1
        return ''.join(rows)

    @staticmethod
    def draw_table_head(label, headers):
        html = f'<thead><tr><th>{label}</th>'
        cell = f'<th>%s</th>'
        html += ''.join([cell % HtmlFormatter.custom_to_html(s) for s in headers])
        html += f'</tr></thead>'
        return html

    @staticmethod
    def draw_table_body(coll, headers, symbol):
        html = f'<tbody>'
        html += HtmlFormatter.draw_rows(coll, headers, symbol)
        html += f'</tbody>'
        return html

    @staticmethod
    def draw_table(t, label, headers, symbol):
        html = f'<table>'
        html += HtmlFormatter.draw_table_head(label, headers)
        html += HtmlFormatter.draw_table_body(t, headers, symbol)
        html += f'</table>'
        return html

    @staticmethod
    def item_to_html(i):
        return '%s%s%s%s%s, { %s }' % (HtmlFormatter.symbol_to_html(i.production.Left), HtmlFormatter.arrow_to_html,
                                        HtmlFormatter.collection_to_html(i.production.Right[:i.pos], ' '),
                                        HtmlFormatter.dot_to_html,
                                        HtmlFormatter.collection_to_html(i.production.Right[i.pos:], ' '),
                                        HtmlFormatter.collection_to_html(i.lookaheads))

    @staticmethod
    def custom_to_html(c):
        if isinstance(c, Symbol):
            return HtmlFormatter.symbol_to_html(c)
        if isinstance(c, Epsilon):
            return HtmlFormatter.epsilon_to_html(c)
        if isinstance(c, Sentence):
            return HtmlFormatter.sentence_to_html(c)
        if isinstance(c, Production):
            return HtmlFormatter.production_to_html(c)
        if isinstance(c, Item):
            return HtmlFormatter.item_to_html(c)
        if isinstance(c, State):
            return HtmlFormatter.custom_to_html(c.state)
        if isinstance(c, tuple):
            return HtmlFormatter.action_to_html(c)
        return f'<span style="color: red"><strong>{c}</strong></span>'

    @staticmethod
    def format_collection(c):
        return [HtmlFormatter.custom_to_html(item) for item in c]

    @staticmethod
    def collection_to_html(c, sep=', ', formatter=None):
        return sep.join([formatter(item) for item in c] if formatter else HtmlFormatter.format_collection(c))

    eol = '\n'

    @staticmethod
    def grammar_to_html(G):
        return f'''<dl>
                <dt><strong>Terminales:</strong></dt> 
                <dd><p>{HtmlFormatter.collection_to_html(G.terminals)}</p></dd>
                <dt><strong>No Terminales:</strong></dt> 
                <dd><p>{HtmlFormatter.collection_to_html(G.nonTerminals)}</p></dd>
                <dt><strong>Producciones:</strong></dt>
                <dd><p>{HtmlFormatter.collection_to_html(G.nonTerminals, '</p><p>', HtmlFormatter.nt_productions_to_html)}</p></dd>
                </dl>'''

    @staticmethod
    def firsts_to_html(G, firsts):
        sf = lambda s: '<p>FIRST(%s) = { %s }</p>' % (HtmlFormatter.symbol_to_html(s), HtmlFormatter.collection_to_html(firsts[s].items()))
        pf = lambda p: '<p>FIRST(%s) = { %s }</p>' % (HtmlFormatter.production_to_html(p), HtmlFormatter.collection_to_html(firsts[p.Right].items()))

        return f'''<dl>
                <dt><strong>No Terminales:</strong></dt> 
                <dd>{HtmlFormatter.collection_to_html(G.nonTerminals, HtmlFormatter.eol, sf)}</dd>
                <dt><strong>Producciones:</strong></dt>
                <dd>{HtmlFormatter.collection_to_html(G.Productions, HtmlFormatter.eol, pf)}</dd>
                </dl>'''

    @staticmethod
    def follows_to_html(G, follows):
        sf = lambda s: '<p>FOLLOW(%s) = { %s }</p>' % (HtmlFormatter.symbol_to_html(s), HtmlFormatter.collection_to_html(follows[s]))

        return f'''<dl>
                <dt><strong>No Terminales:</strong></dt>
                <dd>{HtmlFormatter.collection_to_html(G.nonTerminals, HtmlFormatter.eol, sf)}</dd>
                </dl>'''

    @staticmethod
    def draw_cell(row, symbol):
        return f'''<td {'class="errorCell"' if symbol in row and len(row[symbol]) > 1 else ''} title="{symbol}">%s</td>'''

    @staticmethod
    def cell_class(row, symbol):
        return f'''<td {'class="errorCell"' if symbol in row and len(row[symbol]) > 1 else ''} title="{symbol}">'''

    @staticmethod
    def items_collection_to_html(automaton):
        ni = lambda i: f'<p class="itemCollection">{HtmlFormatter.custom_to_html(i)}</p>'
        nr = lambda n: f'''<table>
                        <thead><tr><th>I<sub>{n.idx}</sub>:</th></tr></thead>
                        <tbody><tr><td>{HtmlFormatter.collection_to_html(n.state, HtmlFormatter.eol, ni)}</td></tr></tbody>
                        </table>'''

        return HtmlFormatter.collection_to_html(automaton, HtmlFormatter.eol, nr)

    @staticmethod
    def action_goto_table_to_html(table, columns, label=''):
        cs = lambda c: f'<th>{HtmlFormatter.symbol_to_html(c)}</th>'
        cl = lambda c: f'<p>{HtmlFormatter.custom_to_html(c)}</p>'
        return f'''<table>
                    <tr><th>{label}</th>{HtmlFormatter.collection_to_html(columns, HtmlFormatter.eol, cs)}</tr>
                    {''.join('<tr>' + f'<th>I<sub>{idx}</sub></th>' + ''.join(HtmlFormatter.cell_class(row, symbol) +
                            (HtmlFormatter.collection_to_html(row[symbol], '', cl) if symbol in row else '-----')  + '</td>'
                                for symbol in columns) + '</tr>' 
                        for idx, row in table.items())}
                    </table>'''

    @staticmethod
    def error_message_to_html(msg):
        return f'<h3 class="error">• {msg}</h3>'

    @staticmethod
    def message_to_html(msg):
        return f'<h3>• {msg}</h3>'