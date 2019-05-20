Vue.config.devtools = true;

const app = new Vue({
  el: "#root",
  data: {
    htmlOutput: "",
    treeOutput: "",
    tree: "",
    pickedTree: "SLR(1)",
    grammar: "",
    render: true
  },
  methods: {
    Analize() {
      eel.pipeline(this.grammar)(response => {
        this.htmlOutput = response;
        console.log("Grammar processed");
        this.render = true;
      });
    },
    Tree() {
      eel.derivationTree(this.grammar, this.tree, this.pickedTree)(response => {
        this.treeOutput = response;
        console.log("String processed");
        this.render = false;
      });
    }
  }
});

// eel.expose(print);
// function print(data){
//     console.log('print');
//     app.print(data);
// }

// setInterval(() => {
//    app.update()
// }, 200);
