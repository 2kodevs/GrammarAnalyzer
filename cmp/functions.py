from cmp.pycompiler import *
from cmp.automata import *
from cmp.utils import *


def read_grammar(text: str):

    def unique(data):
        data = data.copy()
        s = set()
        while True:
            if not data:
                raise StopIteration()
            if data[0] not in s:
                yield data[0]
                s.add(data[0])
            data.pop(0)

    terminals, nonTerminals, productions = [], [], []

    lines = text.split('\n')
    for _ in range(lines.count('')):
        lines.remove('')

    for prod in lines:
        head, sentences = prod.split('-->')

        head, = head.split()
        nonTerminals.append(head)

        for sent in sentences.split('|'):
            productions.append({'Head': head, 'Body': list(sent.split())})
            terminals.extend(productions[-1]['Body'])

    d = dict()
    d['NonTerminals'] = [symbol for symbol in unique(nonTerminals)]
    d['Terminals'] = [symbol for symbol in unique(terminals) if symbol not in nonTerminals and symbol != 'epsilon']
    d['Productions'] = productions

    return Grammar.from_json(json.dumps(d))


def compute_local_first(firsts, alpha):
    first_alpha = ContainerSet()

    try:
        alpha_is_epsilon = alpha.IsEpsilon
    except:
        alpha_is_epsilon = False

    if alpha_is_epsilon:
        first_alpha.set_epsilon()
    else:
        for symbol in alpha:
            first_alpha.update(firsts[symbol])
            if not firsts[symbol].contains_epsilon:
                break
        else:
            first_alpha.set_epsilon()
    return first_alpha


def compute_firsts(G):
    firsts = {}
    change = True

    for terminal in G.terminals:
        firsts[terminal] = ContainerSet(terminal)

    for nonterminal in G.nonTerminals:
        firsts[nonterminal] = ContainerSet()

    while change:
        change = False

        for production in G.Productions:
            X = production.Left
            alpha = production.Right

            first_X = firsts[X]

            try:
                first_alpha = firsts[alpha]
            except:
                first_alpha = firsts[alpha] = ContainerSet()

            local_first = compute_local_first(firsts, alpha)

            change |= first_alpha.hard_update(local_first)
            change |= first_X.hard_update(local_first)
    return firsts


def compute_follows(G, firsts):
    follows = {}
    change = True

    local_firsts = {}

    for nonterminal in G.nonTerminals:
        follows[nonterminal] = ContainerSet()
    follows[G.startSymbol] = ContainerSet(G.EOF)

    while change:
        change = False

        for production in G.Productions:
            X = production.Left
            alpha = production.Right

            follow_X = follows[X]

            for i, Y in enumerate(alpha):
                if Y.IsTerminal:
                    continue
                beta = alpha[i + 1:]
                try:
                    beta_f = local_firsts[beta]
                except KeyError:
                    beta_f = local_firsts[beta] = compute_local_first(firsts, beta)
                change |= follows[Y].update(beta_f)
                if beta_f.contains_epsilon:
                    change |= follows[Y].update(follow_X)

    return follows


def upd_table(table, symbol, trans, val):
    if symbol not in table:
        table[symbol] = {}
    if trans not in table[symbol]:
        table[symbol][trans] = set()
    table[symbol][trans].update([val])
    ans = (len(table[symbol][trans]) == 1)
    return ans


def build_parsing_table(G, firsts, follows):
    M = {}
    ok = True

    for production in G.Productions:
        X = production.Left
        alpha = production.Right

        for t in firsts[alpha]:
            ok &= upd_table(M, X, t, production)

        if firsts[alpha].contains_epsilon:
            for t in follows[X]:
                ok &= upd_table(M, X, t, production)

    return M, ok


def LL1(G):

    firsts = compute_firsts(G)
    follows = compute_follows(G, firsts)

    M, is_LL1 = build_parsing_table(G, firsts, follows)

    return is_LL1, M, firsts, follows


def build_LR0_automaton(G):
    assert len(G.startSymbol.productions) == 1, 'Grammar must be augmented'

    start_production = G.startSymbol.productions[0]
    start_item = Item(start_production, 0)

    automaton = State(start_item, True)

    pending = [start_item]
    visited = {start_item: automaton}

    while pending:
        current_item = pending.pop()
        if current_item.IsReduceItem:
            continue

        # Your code here!!! (Decide which transitions to add)
        transitions = []

        next_item = current_item.NextItem()
        if next_item not in visited:
            visited[next_item] = State(next_item, True)
            pending.append(next_item)
        transitions.append(visited[next_item])

        symbol = current_item.NextSymbol
        if symbol.IsNonTerminal:
            for prod in symbol.productions:
                item = Item(prod, 0)
                if item not in visited:
                    visited[item] = State(item, True)
                    pending.append(item)
                transitions.append(visited[item])

        current_state = visited[current_item]
        # Your code here!!! (Add the decided transitions)
        current_state.add_transition(current_item.NextSymbol.Name, transitions[0])
        for item in transitions[1:]:
            current_state.add_epsilon_transition(item)
    return automaton


class SLR1Parser(ShiftReduceParser):

    def _build_parsing_table(self):
        self.ok = True
        G = self.Augmented = self.G.AugmentedGrammar(True)
        firsts = compute_firsts(G)
        follows = compute_follows(G, firsts)

        self.automaton = build_LR0_automaton(G).to_deterministic(lambda x: "")
        for i, node in enumerate(self.automaton):
            if self.verbose: print(i, node)
            node.idx = i
            node.tag = f'I{i}'

        for node in self.automaton:
            idx = node.idx
            for state in node.state:
                item = state.state
                if item.IsReduceItem:
                    if item.production.Left == G.startSymbol:
                        self.ok &= upd_table(self.action, idx, G.EOF, (SLR1Parser.OK, ''))
                    else:
                        for terminal in follows[item.production.Left]:
                            self.ok &= upd_table(self.action, idx, terminal, (SLR1Parser.REDUCE, item.production))
                else:
                    symbol = item.NextSymbol

                    if symbol.IsTerminal:
                        self.ok &= upd_table(self.action, idx, symbol, (SLR1Parser.SHIFT, node[symbol.Name][0].idx))
                    else:
                        self.ok &= upd_table(self.goto, idx, symbol, node[symbol.Name][0].idx)


def expand(item, firsts):
    next_symbol = item.NextSymbol
    if next_symbol is None or not next_symbol.IsNonTerminal:
        return []

    lookaheads = ContainerSet()
    # Your code here!!! (Compute lookahead for child items)
    for preview in item.Preview():
        lookaheads.hard_update(compute_local_first(firsts, preview))

    assert not lookaheads.contains_epsilon
    # Your code here!!! (Build and return child items)
    return [Item(prod, 0, lookaheads) for prod in next_symbol.productions]


def compress(items):
    centers = {}

    for item in items:
        center = item.Center()
        try:
            lookaheads = centers[center]
        except KeyError:
            centers[center] = lookaheads = set()
        lookaheads.update(item.lookaheads)

    return {Item(x.production, x.pos, set(lookahead)) for x, lookahead in centers.items()}


def closure_lr1(items, firsts):
    closure = ContainerSet(*items)

    changed = True
    while changed:
        changed = False

        new_items = ContainerSet()
        for item in closure:
            new_items.extend(expand(item, firsts))

        changed = closure.update(new_items)

    return compress(closure)


def goto_lr1(items, symbol, firsts=None, just_kernel=False):
    assert just_kernel or firsts is not None, '`firsts` must be provided if `just_kernel=False`'
    items = frozenset(item.NextItem() for item in items if item.NextSymbol == symbol)
    return items if just_kernel else closure_lr1(items, firsts)


def build_LR1_automaton(G):
    assert len(G.startSymbol.productions) == 1, 'Grammar must be augmented'

    firsts = compute_firsts(G)
    firsts[G.EOF] = ContainerSet(G.EOF)

    start_production = G.startSymbol.productions[0]
    start_item = Item(start_production, 0, lookaheads=(G.EOF,))
    start = frozenset([start_item])

    closure = closure_lr1(start, firsts)
    automaton = State(frozenset(closure), True)

    pending = [start]
    visited = {start: automaton}

    while pending:
        current = pending.pop()
        current_state = visited[current]

        for symbol in G.terminals + G.nonTerminals:
            # Your code here!!! (Get/Build `next_state`)
            items = current_state.state
            kernel = goto_lr1(items, symbol, just_kernel=True)
            if not kernel:
                continue
            try:
                next_state = visited[kernel]
            except KeyError:
                closure = goto_lr1(items, symbol, firsts)
                next_state = visited[kernel] = State(frozenset(closure), True)
                pending.append(kernel)

            current_state.add_transition(symbol.Name, next_state)

    automaton.set_formatter(lambda x: "")
    return automaton


class LR1Parser(ShiftReduceParser):
    def _build_parsing_table(self):
        self.ok = True
        G = self.Augmented = self.G.AugmentedGrammar(True)

        automaton = self.automaton = build_LR1_automaton(G)
        for i, node in enumerate(automaton):
            if self.verbose: print(i, '\t', '\n\t '.join(str(x) for x in node.state), '\n')
            node.idx = i
            node.tag = f'I{i}'

        for node in automaton:
            idx = node.idx
            for item in node.state:
                if item.IsReduceItem:
                    prod = item.production
                    if prod.Left == G.startSymbol:
                        self.ok &= upd_table(self.action, idx, G.EOF, (ShiftReduceParser.OK, ''))
                    else:
                        for lookahead in item.lookaheads:
                            self.ok &= upd_table(self.action, idx, lookahead, (ShiftReduceParser.REDUCE, prod))
                else:
                    next_symbol = item.NextSymbol
                    if next_symbol.IsTerminal:
                        self.ok &= upd_table(self.action, idx, next_symbol, (ShiftReduceParser.SHIFT, node[next_symbol.Name][0].idx))
                    else:
                        self.ok &= upd_table(self.goto, idx, next_symbol, node[next_symbol.Name][0].idx)


def mergue_items_lookaheads(items, others):
    if len(items) != len(others):
        return False

    new_lookaheads = []
    for item in items:
        for item2 in others:
            if item.Center() == item2.Center():
                new_lookaheads.append(item2.lookaheads)
                break
        else:
            return False

    for item, new_lookahead in zip(items, new_lookaheads):
        item.lookaheads = item.lookaheads.union(new_lookahead)

    return True


def build_LALR1_automaton(G):
    lr1_automaton  = build_LR1_automaton(G)
    states = list(lr1_automaton)
    new_states = []
    visited = {}

    for i, state in enumerate(states):
        if state not in visited:
            # creates items
            items = [item.Center() for item in state.state]

            # check for states with same center
            for state2 in states[i:]:
                if mergue_items_lookaheads(items, state2.state):
                    visited[state2] = len(new_states)

            # add new state
            new_states.append(State(frozenset(items), True))

    # making transitions
    for state in states:
        new_state = new_states[visited[state]]
        for symbol, transitions in state.transitions.items():
            for state2 in transitions:
                new_state2 = new_states[visited[state2]]
                # check if the transition already exists
                if symbol not in new_state.transitions or new_state2 not in new_state.transitions[symbol]:
                    new_state.add_transition(symbol, new_state2)

    new_states[0].set_formatter(empty_formatter)
    return new_states[0]


class LALR1Parser(ShiftReduceParser):
    def _build_parsing_table(self):
        self.ok = True
        G = self.Augmented = self.G.AugmentedGrammar(True)

        automaton = self.automaton = build_LALR1_automaton(G)
        for i, node in enumerate(automaton):
            if self.verbose: print(i, '\t', '\n\t '.join(str(x) for x in node.state), '\n')
            node.idx = i
            node.tag = f'I{i}'

        for node in automaton:
            idx = node.idx
            for item in node.state:
                if item.IsReduceItem:
                    prod = item.production
                    if prod.Left == G.startSymbol:
                        self.ok &= upd_table(self.action, idx, G.EOF, (ShiftReduceParser.OK, ''))
                    else:
                        for lookahead in item.lookaheads:
                            self.ok &= upd_table(self.action, idx, lookahead, (ShiftReduceParser.REDUCE, prod))
                else:
                    next_symbol = item.NextSymbol
                    if next_symbol.IsTerminal:
                        self.ok &= upd_table(self.action, idx, next_symbol, (ShiftReduceParser.SHIFT, node[next_symbol.Name][0].idx))
                    else:
                        self.ok &= upd_table(self.goto, idx, next_symbol, node[next_symbol.Name][0].idx)


def is_null(G):

    def dfs(symbol):
        visited.update(symbol)
        if not isinstance(symbol, Terminal):
            for production in symbol.productions:
                for s in production.Right:
                    if s not in visited:
                        dfs(s)
                if all(s in not_null_symbol for s in production.Right):
                    not_null_symbol.update([s])

    not_null_symbol = set([t for t in G.terminals])
    visited = set([t for t in G.terminals])
    not_null_symbol.add(G.EOF)
    visited.add(G.EOF)

    dfs(G.startSymbol)

    return G.startSymbol in not_null_symbol


def without_recursion(G):
    G.Productions = []

    nonTerminals = G.nonTerminals.copy()
    for nt in nonTerminals:
        bad_prod = [Sentence(*prod.Right[1:]) for prod in nt.productions if len(prod.Right) > 0 and prod.Right[0] == nt]
        good_prod = [Sentence(*prod.Right) for prod in nt.productions if len(prod.Right) == 0 or prod.Right[0] != nt]

        if len(bad_prod) > 0:
            nt.productions = []
            s_new = G.NonTerminal(f"{nt.Name}<sub>0</sub>")

            for prod in good_prod:
                nt %= prod + s_new

            for prod in bad_prod:
                s_new %= prod + s_new

            s_new %= G.Epsilon
        else:
            G.Productions.extend(nt.productions)


def without_common_prefix(G: Grammar):
    G.Productions = []
    number = {nt.Name: 1 for nt in G.nonTerminals}

    pending = G.nonTerminals.copy()
    while pending:
        nt = pending.pop()
        prod = nt.productions.copy()
        nt.productions = []
        solved = []

        for i, p1 in enumerate(prod):
            if p1 not in solved:
                solved.append(p1)
                common = [p1]
                prefix = len(p1.Right)

                for p2 in prod[i + 1:]:
                    sz = 0
                    for x, y in zip(p1.Right, p2.Right):
                        if x == y:
                            sz += 1
                        else:
                            break
                    if sz > 0:
                        solved.append(p2)
                        common.append(p2)
                        prefix = min(prefix, sz)

                if len(common) > 1:
                    try:
                        name = nt.Name[:nt.Name.index('<sub>')]
                    except:
                        name = nt.Name
                    s_new = G.NonTerminal(f'{name}<sub>{number[name]}</sub>')
                    pending.append(s_new)
                    number[name] += 1

                    nt %= Sentence(*p1.Right[:prefix]) + s_new
                    for p3 in common:
                        if len(p3.Right) == prefix:
                            s_new %= G.Epsilon
                        else:
                            s_new %= Sentence(*p3.Right[prefix:])
                else:
                    nt %= p1.Right


def derivation_tree(d):

    def add_trans(cur, transitions):
        for symbol in transitions:
            if symbol.IsTerminal:
                cur.add_transition('', State(symbol, True))
            else:
                s = State(symbol, True)
                try:
                    old[symbol].append(s)
                except KeyError:
                    old[symbol] = [s]
                cur.add_transition('', s)
        if len(transitions) == 0:
            cur.add_transition('', State(transitions, True))

    p1 = d[0]
    old = {}
    root = State(p1.Left.Name, True)
    add_trans(root, p1.Right)

    for p in d[1:]:
        node = old[p.Left].pop()
        add_trans(node, p.Right)

    return root


def verify_regularity(G):

    def check(prod, epsilon):
        sz = len(prod)
        if sz == 0:
            return epsilon
        if sz == 1:
            return prod[0].IsTerminal
        if sz == 2:
            return prod[0].IsTerminal and prod[1].IsNonTerminal
        return False

    regular = all([check(p.Right, p.Left == G.startSymbol) for p in G.Productions])

    return regular


def grammarToNFA(G):
    states = {}
    finals = []
    transitions = {}

    for var in G.nonTerminals:
        states[var] = len(states)

    for p in G.Productions:
        lastTerminal = p.Right[0]
        lastState = states[p.Left]

        for i, var in enumerate(p.Right):
            if i == len(p.Right) - 1:
                if var in G.terminals:
                    l = len(states)
                    states[str(var) + f'{len(states)}'] = l
                    if l not in finals:
                        finals.append(l)
                    try:
                        transitions[lastState, lastTerminal].append(l)
                    except KeyError:
                        transitions[lastState, lastTerminal] = [l]
                else:
                    try:
                        transitions[lastState, lastTerminal].append(states[var])
                    except KeyError:
                        transitions[lastState, lastTerminal] = [states[var]]
            lastTerminal = var

    return NFA(len(states), finals, transitions, start=states[G.startSymbol])


def regex_union(regex, other):
    if regex is None:
        return other

    if other is None:
        return regex

    if regex == other:
        return regex

    return f'({regex}|{other})'


def regex_concat(regex, other):
    if regex is None or other is None:
        return None

    if regex == 'ε':
        return other

    if other == 'ε':
        return regex

    return f'{regex}{other}'


def regex_star(regex):
    if regex is None or regex == 'ε':
        return regex

    return f'({regex})*'


def NFAToRegex(automaton: NFA):
    B = ['ε' if i in automaton.finals else None for i in range(automaton.states)]
    A = [[None for j in range(automaton.states)] for i in range(automaton.states)]

    for i in range(automaton.states):
        A[i][i] = 'ε'

    for i in range(automaton.states):
        for var in automaton.vocabulary:
            try:
                dest = automaton.transitions[i][var]
                for d in dest:
                    if A[i][d] == None or A[i][d] == 'ε':
                        A[i][d] = str(var)
                    else:
                        A[i][d] = regex_union(A[i][d], str(var))
            except KeyError:
                continue

    for n in range(automaton.states - 1, -1, -1):
        B[n] = regex_concat(regex_star(A[n][n]), B[n])
        for j in range(n):
            A[n][j] = regex_concat(regex_star(A[n][n]), A[n][j])
        for i in range(n):
            B[i] = regex_union(B[i], regex_concat(A[i][n], B[n]))
            for j in range(n):
                A[i][j] = regex_union(A[i][j], regex_concat(A[i][n], A[n][j]))

    return B[0]


def regex_analizer(G):
    ok = verify_regularity(G)
    if ok:
        automaton = grammarToNFA(G)
        regex =  NFAToRegex(automaton)
        print(regex)
        return ok, State.from_nfa(automaton), regex
    else:
        return ok, 'empty', 'empty'


def ll1_conflict(G: Grammar, table):
    queue = [([G.startSymbol], None, '', False)]

    def enqueue(der, ter, word, conflict, data):
        conflict = conflict or len(data) > 1
        for prod in data:
            adv = der.copy()
            adv[:1] = [s for s in prod.Right]
            queue.append((adv, ter, word, conflict))

    while queue:
        der, ter, word, conflict = queue.pop(0)
        if ter:
            if ter == G.EOF:
                if conflict:
                    return word
                continue
            if der[0] != ter:
                enqueue(der, ter, word, conflict, table[der[0]][ter])
                continue
            der.pop(0)
        if not der:
            if conflict:
                return word
            continue
        for symbol in table[der[0]]:
            enqueue(der, symbol, word + str(symbol), conflict, table[der[0]][symbol])


def action_goto_conflict(action, goto):
    queue = [([0], None, '', False)]

    def go(stack, ter, word, conflict, data):
        conflict = conflict or len(data) > 1
        for move in data:
            queue.append((stack + [move], ter, word, conflict))

    def enqueue(stack, ter, word, conflict, data):
        conflict = conflict or len(data) > 1
        reduce = [cell[1] for cell in data if cell[0] == SLR1Parser.REDUCE]
        shift = [cell[1] for cell in data if cell[0] == SLR1Parser.SHIFT]
        for prod in reduce:
            new_stack = stack.copy()
            if len(prod.Right):
                new_stack = new_stack[:-len(prod.Right)]
            go(new_stack, ter, word, conflict, goto[new_stack[-1]][prod.Left])
        for s in shift:
            queue.append((stack + [s], None, word + str(ter), conflict))

    while queue:
        stack, ter, word, conflict = queue.pop(0)
        state = stack[-1]
        if ter:
            try:
                if any([cell for cell in action[state][ter] if cell[0] == SLR1Parser.OK]) and conflict:
                    return word
                enqueue(stack, ter, word, conflict, action[state][ter])
            except Exception as e:
                print(e)
        else:
            for symbol in action[state]:
                queue.append((stack.copy(), symbol, word, conflict))


# def Parse(parser_class, G):
#     parser = parser_class(G)
#     html = []
#     html.append(html.items_collection_to_html(parser.automaton))
#     html.append(parser.automaton._repr_svg_())
#     html.append(html.draw_table(action, 'ACTION', parser.Augmented.terminals + [parser.Augmented.EOF], 'I<sub>%s</sub>'))
#     html.append(html.draw_table(goto, 'GOTO', parser.Augmented.nonTerminals, 'I<sub>%s</sub>'))


