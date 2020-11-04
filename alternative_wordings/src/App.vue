<template>
  <div id="app">
    <!--Header, input text area, and button section-->
    <div class="form-group">
      <h2 style="color: black">Exploring alternative wordings</h2>
      <textarea
        id="userenglish"
        name="text"
        v-model="inputText"
        rows="4"
        cols="50"
        required
      >
      </textarea
      ><br /><br />
      <button
        class="continue"
        @click="
          getAlts(inputText);
          incremental(inputText, '', false);
          showResults = true;
        "
      >
        Continue
      </button>
    </div>
    <!--Area where you can switch out words in the sentence with most likely alternatives-->
    <div class="focus-sentence">
      <span class="tooltip" v-for="(word, ind) in incrementalData.tokens">
        <!--If you click on a word to replace it make the replacement yellow-->
        <span v-if="ind == selectedIdx" style="background-color: yellow">{{
          word
        }}</span>
        <span v-else>{{ word }}</span>
        <!--Drop down selector for alternative words from prediction-->
        <div id="ind" class="tooltiptext">
          <span v-for="i in incrementalData.predictions[ind]">
            <button
              class="plain"
              @click="
                recalculate(i, ind);
                selectedIdx = ind;
              "
            >
              {{ i }}</button
            ><br />
          </span>
        </div>
      </span>
    </div>

    <!--Alternative sentences for original sentence with parts highlighted-->
    <div v-if="showResults" class="results">
      <br />
      <ul>
        <li v-for="(set, idx) in altsData.colorCoding" class="tooltip">
          <button v-bind:id="idx" @click="toggleShowing($event)" class="plain">
            <span v-if="isShowing[idx]">-</span>
            <span v-else>+</span>
          </button>
          <button
            @click="incremental(altsData.alternatives[idx][0], '', false)"
            style="font-size: 15px"
            class="plain"
          >
            <span
              v-for="chunk in set[0]"
              v-bind:style="{ 'background-color': colors[chunk[1]] }"
            >
              {{ chunk[0] }}
            </span>
          </button>
          <br />
          <ul v-show="isShowing[idx]">
            <li v-for="(alt, subidx) in restOfSet(set)">
              <button
                @click="
                  incremental(altsData.alternatives[idx][subidx + 1], '', false)
                "
                style="font-size: 15px"
                class="plain"
              >
                <span v-for="chunk in alt">
                  {{ chunk[0] }}
                </span>
              </button>
            </li>
          </ul>
        </li>
      </ul>
    </div>
    <!--Alternative sentences displayed once a word is changed -->
    <div v-else class="alterations">
      <p style="font-size: 20px">{{ withChangedWord }}</p>
      <div class="grid-container">
        <div class="grid-item" style="text-align: right; white-space: nowrap">
          {{ prefix }}
        </div>
        <div class="grid-item" style="color: #666666">
          <div v-for="(completion, optionidx) in completions">
            <button
              @click="incremental(fulltext(prefix, completion), '', false)"
              style="font-size: 20px; color: #666666"
              class="plain"
            >
              <!--Bold words which are different between alternative and original-->
              <span v-for="(word, wordidx) in completion">
                <span
                  v-if="isDifferent(word, optionidx, wordidx)"
                  class="diff"
                  >{{ word }}</span
                >
                <span v-else>{{ word }}</span>
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      isShowing: [false],
      inputText: "",
      showResults: true,
      altsData: { alternatives: [], colorCoding: [], test: true },
      colors: [
        "white",
        "#ABE3BB",
        "#E4B0AF",
        "#3DA1B8",
        "#E4E2AF",
        "#B8473D",
        "#B39FCF",
        "#30915F",
      ],
      incrementalData: {
        final: "",
        expected: "",
        tokens: [],
        predictions: [],
        score: 0,
      },
      withChangedWord: "",
      prefix: "",
      completions: [],
      differences: [],
      selectedIdx: -1,
    };
  },

  methods: {
    restOfSet(group) {
      console.log(group);
      return group.slice(1);
    },
    toggleShowing: function (event) {
      var targetId = event.currentTarget.id;
      console.log(targetId);
      if (this.isShowing[targetId]) {
        this.$set(this.isShowing, targetId, false);
      } else {
        {
          this.$set(this.isShowing, targetId, true);
        }
      }
      console.log(this.isShowing[targetId]);
    },
    handle(idx) {
      console.log(idx);
      return this.isShowing[idx];
    },
    isDifferent(string, optionidx, wordidx) {
      // Used to bold different words in alternative sentences

      console.log(string);
      console.log(optionidx);
      console.log(wordidx);
      if (this.differences[optionidx].indexOf(wordidx) > -1) {
        console.log("true");
        return true;
      } else {
        return false;
      }
    },
    async getAlts(inputText) {
      // Gets alternatives for inputText along with chunks for color coding
      // Used with continue button

      var url = new URL("/api/result", window.location);
      var params = {
        english: inputText,
      };
      url.searchParams.append("q", JSON.stringify(params));
      const res = await fetch(url);
      const input = await res.json();
      this.altsData = input;
    },

    async incremental(inputText, prefix, recalculation) {
      // Fills this.incrementalData with inputText, expected machine translation.
      // list of tokens in final sequence, list of top 10 alternative words for each token,
      // and a score for average predictability.

      var url = new URL("/api/incremental", window.location);
      var params = {
        english: inputText,
        prefix: prefix,
        recalculation: recalculation,
      };
      url.searchParams.append("q", JSON.stringify(params));
      const res = await fetch(url);
      const input = await res.json();
      this.incrementalData = input;
      // reset selectedIdx in case this was called through selecting alternative word
      this.selectedIdx = -1;
    },

    async recalculate(changedword, index) {
      // Get alternative endings once word is chosen from word probability list

      this.$set(this.incrementalData.tokens, index, changedword);
      // get sentence up to and including changed word
      var thelist = this.incrementalData.tokens.slice(0, index + 1);
      var newinputstr = thelist.join("").replace(/\u00a0/g, " ");
      // set as prefix
      this.prefix = newinputstr;
      // this.incremental(this.inputText, newinputstr, true)
      // this.withChangedWord = this.incrementalData.tokens.join('').replace(/\u00a0/g, ' ');

      var url = new URL("/api/completion", window.location);
      var params = {
        sentence: this.inputText,
        prefix: newinputstr,
      };
      url.searchParams.append("q", JSON.stringify(params));
      const res = await fetch(url);
      const input = await res.json();
      let tolist = [];
      var x;
      for (x in input.endings) {
        //split each string into words, preserving spaces
        tolist.push(
          input.endings[x].split(/(\S+\s+)/).filter(function (n) {
            return n;
          })
        );
      }
      this.completions = tolist;
      this.differences = input.differences;

      this.showResults = false;
    },

    fulltext(prefix, completion) {
      return prefix + completion.join("");
    },
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

.diff {
  font-weight: bold;
}

ul {
  line-height: 300%;
}

.form-group {
  background-color: lightgray;
  padding: 3%;
}
.focus-sentence {
  background-color: #eeeeee;
  font-size: 25px;
  padding: 5%;
}
.results {
  text-align: left;
  margin-left: 10%;
  margin-right: 10%;
}
.tooltip {
  position: relative;
  display: inline-block;
}
.tooltip .tooltiptext {
  visibility: hidden;
  /* width: 150px; */
  background-color: white;
  color: white;
  text-align: left;
  border-radius: 6px;
  font-size: 66.67%;
  position: absolute;
  z-index: 1;
}
.tooltiptext {
  padding: 10%;
}
.tooltiptext button.plain {
  font-size: 20px;
  margin: 0.1em;
}
.tooltip:hover .tooltiptext {
  visibility: visible;
}
.tooltip:hover {
  cursor: context-menu;
}

button.plain {
  background: none;
  border: none;
  text-align: left;
  white-space: nowrap;
}
button.plain:hover {
  cursor: pointer;
}

button.plain:focus {
  outline: none;
}

.continue {
  box-shadow: inset 0px 1px 0px 0px white;
  background: linear-gradient(to bottom, white 5%, #f6f6f6 100%);
  background-color: white;
  border-radius: 6px;
  border: 1px solid #dcdcdc;
  display: inline-block;
  color: #666666;
  font-size: 15px;
  font-weight: bold;
  padding: 6px 24px;
  text-shadow: 0px 1px 0px white;
}
.continue:hover {
  background: linear-gradient(to bottom, #f6f6f6 5%, white 100%);
  background-color: #f6f6f6;
  cursor: pointer;
}
.grid-container {
  display: grid;
  grid-template-columns: auto auto auto;
  padding: 10px;
}
.grid-item {
  font-size: 20px;
  text-align: left;
  padding: 0px;
}

.results .tooltip {
  vertical-align: top;
}
</style>
