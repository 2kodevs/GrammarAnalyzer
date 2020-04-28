# GrammarAnalyzer

This project analyzes grammar and performs various processes such as:

* If possible, generate parsers LL(1), SLR(1), LR(1), LALR(1) for grammar.
* Build the derivation trees for the aforementioned parsers, as well as automaton and related tables.
* Shows possible conflict strings for generated parsers.
* Analyze if the grammar is regular, and if so, build an automaton and regular expression from the grammar.

## Starting
To use the project, clone it or download it to your local computer.


### Requirements 📋
It is necessary to have `python v-3.7.2`, [pydot](https://pypi.org/project/pydot/), [eel](https://github.com/ChrisKnott/Eel), `graphviz` and` chrome` or `chromium` installed to make full use of the application's functionalities.

Is recommended to install python and later use `make install` to install the other dependencies


### Installation 🔧

To execute the project, just open the console from the root location of the project and execute:

```
python3 main.py
```
Is your `python3` installation is accesible using `python` you can use too the command
```
make
```

It is important to note that the grammar must be inserted in the form `E --> a` where `E` is `a` **_nonterminal_** that produces a **_terminal_** `a`. Any symbol appearing in the left part of a production will be interpreted as a **_nonterminal_** one.

To insert a string of the language and analyze if it belongs to this, and to build a derivation tree of a certain parser generated from this string, it is imperative that all the symbols of the string be separated by a blank space.

## Authors ✒️

* **Lazaro Raul Iglesias Vera** -- [stdevRulo](https://github.com/stdevRulo)
* **Miguel Tenorio Potrony** - [stdevAntiD2ta](https://github.com/stdevAntiD2ta)

## License 📄

This project is under the License (MIT License) - see the file [LICENSE.md](LICENSE.md) for details.
