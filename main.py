import eel
import logging
from core.cmp.functions import analize_grammar, make_tree


@eel.expose
def pipeline(data):
    values = analize_grammar(data)
    fd = open("./web/template.html", 'r', encoding='UTF-8')
    data = fd.read()
    fd.close()

    sec = data.split('%s')
    html = []
    for i in range(len(sec)):
        html.append(sec[i])
        html.append(values[i])

    return ''.join(html)

@eel.expose
def derivationTree(text, w, parser_name):
    d = make_tree(text, w, parser_name)
    return d

def main():
    eel.init('web')

    eel_options = {'port': 8045}
    eel.start('index.html', size=(1000, 860), options=eel_options, block=False, suppress_error=True)

    while True:
        eel.sleep(0.1)


if __name__ == '__main__':
    main()