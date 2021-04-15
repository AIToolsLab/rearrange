from fairseq.token_generation_constraints import pack_constraints
from fairseq.models.transformer import TransformerModel
import re


class mbartAlt:
    def __init__(self, lang: str):
        self.bart = TransformerModel.from_pretrained(
            "mbart50.ft.nn",
            checkpoint_file="model.pt",
            data_name_or_path="mbart50.ft.nn",
            bpe="sentencepiece",
            sentencepiece_model="mbart50.ft.nn/sentence.bpe.model",
            lang_dict="mbart50.ft.nn/ML50_langs.txt",
            target_lang=lang,
            source_lang="en_XX",
            encoder_langtok="src",
        )
        self.bart.eval()
        self.lang = lang

    def constraint2tensor(self, constraints: [str]):
        for i, constraint_list in enumerate(constraints):
            constraints[i] = [
                # encode with src_dict as this becomes tgt
                self.bart.src_dict.encode_line(
                    self.bart.apply_bpe(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]
        return pack_constraints(constraints)

    def clean_lang_tok(self, input: str):
        return re.sub("^[\[].*[\]] ", "", input)

    def sample(self, sentence, beam, verbose, **kwargs):
        tokenized_sentence = [self.bart.encode(sentence)]
        hypos = self.bart.generate(tokenized_sentence, beam, verbose, **kwargs)[0]

        return hypos

    def round_trip(self, sentence: str, constraints: [str]):
        constraints_tensor = self.constraint2tensor([constraints])
        away = self.bart.translate(sentence)
        away = self.clean_lang_tok(away)
        # switch translation direction
        orig_tgt = self.bart.task.args.target_lang
        orig_src = self.bart.task.args.source_lang
        self.bart.task.args.target_lang = orig_src
        self.bart.task.args.source_lang = orig_tgt

        returned = self.sample(
            away,
            beam=5,
            verbose=True,
            constraints="ordered",
            inference_step_args={"constraints": constraints_tensor},
        )
        resultset = []
        for i in range(3):
            resultset.append(
                (
                    returned[i]["score"],
                    self.clean_lang_tok(self.bart.decode(returned[i]["tokens"])),
                )
            )
        # restore original translation direction
        self.bart.task.args.target_lang = orig_tgt
        self.bart.task.args.source_lang = orig_src

        # return self.clean_lang_tok(returned)
        return resultset

    def get_prefix_alts(self, sentence, prefixes: [str]):
        return [self.round_trip(sentence, [prefix]) for prefix in prefixes]


if __name__ == "__main__":
    mbart = mbartAlt("de_DE")
    print(
        mbart.get_prefix_alts(
            "She shot the cow during a time of scarcity to feed her hungry family.",
            [
                "During a time of scarcity",
                "Of scarcity",
                "She ",
                "The cow",
                "Her hungry family",
                "To feed her hungry family",
                "She shot",
            ],
        )
    )
