#!/usr/bin/env python3
"""Build the grammar-coaching evaluation dataset (prompts.json)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _corr(
    eid: str,
    category: str,
    difficulty: str,
    source: str,
    ground_truth: str,
    acceptable: list[str],
    explanation: str,
    key_requirements: list[str],
    must_not_contain: list[str],
    meaning_tokens: list[str],
    extra: dict | None = None,
) -> dict:
    acc = list(dict.fromkeys([ground_truth, *acceptable]))
    item = {
        "id": eid,
        "task": "correct_sentence",
        "category": category,
        "difficulty": difficulty,
        "source_sentence": source,
        "prompt": (
            "Correct the English below. Output ONLY the corrected sentence. "
            "Preserve the original meaning. If it is already correct, repeat it unchanged. "
            "No quotes, no labels, no explanation.\n\n"
            f"{source}"
        ),
        "ground_truth": ground_truth,
        "acceptable_answers": acc,
        "explanation": explanation,
        "key_requirements": key_requirements,
        "must_not_contain": must_not_contain,
        "meaning_tokens": meaning_tokens,
        "error_tokens": must_not_contain,
    }
    if extra:
        item.update(extra)
    return item


def _judge_complete(eid, difficulty, stem, child, sample, expected, explanation, keys) -> dict:
    return {
        "id": eid,
        "task": "judge_complete",
        "category": "judge_completion",
        "difficulty": difficulty,
        "prompt": (
            "You are a kind tutor for kids ages 5-8. They are finishing a sentence. "
            "sample_ending is only ONE possible answer. Accept any child_said that "
            "makes a real, kid-safe English sentence with the stem. "
            "Reject nonsense, empty guesses, or words that do not finish the thought. "
            'JSON only: {"correct": true/false}\n\n'
            f"stem: {stem}\nchild_said: {child}\nsample_ending: {sample}"
        ),
        "stem": stem,
        "child_said": child,
        "sample_ending": sample,
        "ground_truth": {"correct": expected},
        "acceptable_answers": [{"correct": expected}],
        "explanation": explanation,
        "key_requirements": keys,
        "must_not_contain": [],
        "meaning_tokens": [],
    }


def _judge_mistake(eid, difficulty, spoken, child, sample, expected, explanation, keys) -> dict:
    return {
        "id": eid,
        "task": "judge_mistake",
        "category": "judge_mistake",
        "difficulty": difficulty,
        "prompt": (
            "You are a kind tutor for kids ages 5-8. They must fix one error in a sentence. "
            "sample_fix is one good correction. Accept paraphrases that fix the SAME error. "
            "Reject repeating the error or a totally different sentence. "
            'JSON only: {"correct": true/false}\n\n'
            f"spoken: {spoken}\nchild_said: {child}\nsample_fix: {sample}"
        ),
        "spoken": spoken,
        "child_said": child,
        "sample_fix": sample,
        "ground_truth": {"correct": expected},
        "acceptable_answers": [{"correct": expected}],
        "explanation": explanation,
        "key_requirements": keys,
        "must_not_contain": [],
        "meaning_tokens": [],
    }


def _invent(eid, kind, banned, extra_instruction: str) -> dict:
    if kind == "complete":
        prompt = (
            "Invent one brand-new sentence-completion prompt for kids ages 5-8. "
            "Kid-safe everyday English. Never copy a banned stem. "
            "Stem is 6-12 words that stop mid-thought. expected is one SAMPLE noun ending, not the only answer. "
            "Do not end the stem on a lonely adjective. "
            'JSON only: {"stem":"...","expected":"...","full":"..."}\n\n'
            f"banned_stems: {json.dumps(banned)}"
        )
        keys = ["stem", "expected", "full", "json_only", "kid_safe"]
        cat = "invent_complete"
    else:
        prompt = (
            "Invent one SHORT spoken sentence (6 to 10 words) with exactly one kid-friendly error. "
            "Use only: don't/doesn't, was/were, goed/went, have/has, see/sea, free/three, or sink/think. "
            "No silly or surreal scenes. spoken and correct must be almost the same except that one error. "
            'JSON only: {"spoken":"...","correct":"...","kind":"grammar|pronunciation","hint":"..."}\n\n'
            f"{extra_instruction}"
        )
        keys = ["spoken", "correct", "kind", "hint", "json_only", "single_error"]
        cat = "invent_mistake"
    return {
        "id": eid,
        "task": f"invent_{kind}",
        "category": cat,
        "difficulty": "hard",
        "prompt": prompt,
        "banned_stems": banned,
        "ground_truth": {"schema": kind},
        "acceptable_answers": [],
        "explanation": "Score structural validity, kid-safety, and instruction following, not a single string.",
        "key_requirements": keys,
        "must_not_contain": banned if kind == "complete" else [],
        "meaning_tokens": [],
    }


def build() -> dict:
    items: list[dict] = []

    dont_pairs = [
        ("She", "apples", "easy"),
        ("He", "broccoli", "easy"),
        ("My sister", "loud music", "medium"),
        ("The teacher", "messy desks", "medium"),
        ("That dog", "thunder", "easy"),
        ("The baby", "cold peas", "easy"),
        ("Our neighbor", "late buses", "medium"),
        ("The captain", "rough waves", "hard"),
        ("Everybody", "waiting in line", "hard"),
        ("Each player", "unfair rules", "hard"),
        ("Nobody", "spoiled milk", "hard"),
        ("The class", "pop quizzes", "medium"),
        ("My cousin", "spicy soup", "easy"),
        ("The nurse", "noisy halls", "medium"),
        ("This cat", "water", "easy"),
        ("The mayor", "traffic jams", "hard"),
        ("Her friend", "scary movies", "easy"),
        ("The chef", "burnt toast", "medium"),
        ("That bird", "cages", "easy"),
        ("The librarian", "torn pages", "medium"),
    ]
    for i, (subj, obj, diff) in enumerate(dont_pairs, 1):
        src = f"{subj} don't like {obj}."
        gt = f"{subj} doesn't like {obj}."
        alt = f"{subj} does not like {obj}."
        items.append(
            _corr(
                f"sva-dont-{i:03d}",
                "subject_verb_agreement",
                diff,
                src,
                gt,
                [alt],
                f"Third-person singular requires doesn't/does not, not don't.",
                ["doesn't or does not", "preserve like", obj.split()[0].lower()],
                ["don't"],
                [w.lower() for w in (subj.split()[-1], "like", *obj.split())],
            )
        )

    have_pairs = [
        ("He", "two cats", "has", "easy"),
        ("She", "a red bike", "has", "easy"),
        ("It", "four legs", "has", "easy"),
        ("My brother", "new shoes", "has", "easy"),
        ("The dog", "a loud bark", "has", "easy"),
        ("They", "three tickets", "have", "easy"),
        ("We", "plenty of time", "have", "easy"),
        ("You", "a kind smile", "have", "easy"),
        ("The children", "art class today", "have", "medium"),
        ("Everyone", "a nametag", "has", "hard"),
        ("Each student", "a pencil", "has", "hard"),
        ("The team", "a new coach", "has", "medium"),
        ("Those birds", "sharp beaks", "have", "easy"),
        ("This soup", "too much salt", "has", "medium"),
        ("People", "different hobbies", "have", "medium"),
        ("Somebody", "my umbrella", "has", "hard"),
        ("The news", "a surprise ending", "has", "hard"),
        ("Mathematics", "many rules", "has", "hard"),
        ("My glasses", "a scratch", "have", "hard"),
        ("The scissors", "a loose screw", "have", "hard"),
    ]
    for i, (subj, obj, verb, diff) in enumerate(have_pairs, 1):
        wrong = "have" if verb == "has" else "has"
        src = f"{subj} {wrong} {obj}."
        gt = f"{subj} {verb} {obj}."
        items.append(
            _corr(
                f"sva-have-{i:03d}",
                "subject_verb_agreement",
                diff,
                src,
                gt,
                [gt],
                f"Agreement: {subj} takes {verb}.",
                [verb, obj.split()[0].lower()],
                [wrong] if wrong != verb else [],
                [w.lower() for w in (subj.split()[-1], *obj.split())],
            )
        )

    was_were = [
        ("We", "playing tag", "were", "easy"),
        ("They", "eating lunch", "were", "easy"),
        ("You", "very quiet", "were", "easy"),
        ("I", "tired yesterday", "was", "easy"),
        ("She", "at the library", "was", "easy"),
        ("The kids", "building a fort", "were", "medium"),
        ("My parents", "in the kitchen", "were", "medium"),
        ("The team", "ready to start", "was", "hard"),
        ("There", "three cookies left", "were", "hard"),
        ("There", "a cat on the sofa", "was", "medium"),
        ("The scissors", "on the table", "were", "hard"),
        ("Everyone", "excited", "was", "hard"),
        ("Both doors", "locked", "were", "medium"),
        ("Neither answer", "correct", "was", "hard"),
        ("A flock of birds", "overhead", "was", "hard"),
        ("The news", "surprising", "was", "hard"),
        ("You and I", "late", "were", "medium"),
        ("The pair of shoes", "too small", "was", "hard"),
        ("My family", "home last night", "was", "hard"),
        ("Those books", "on the floor", "were", "easy"),
    ]
    for i, (subj, rest, verb, diff) in enumerate(was_were, 1):
        wrong = "was" if verb == "were" else "were"
        src = f"{subj} {wrong} {rest}."
        gt = f"{subj} {verb} {rest}."
        items.append(
            _corr(
                f"sva-be-{i:03d}",
                "subject_verb_agreement",
                diff,
                src,
                gt,
                [gt],
                f"{subj} agrees with {verb}.",
                [verb],
                [wrong],
                [w.lower() for w in (subj.split()[-1], *rest.split())],
            )
        )

    tense = [
        ("Yesterday I goed to the park.", "Yesterday I went to the park.", ["Yesterday I went to the park."], "goed", "verb_tense", "easy", ["yesterday", "went", "park"]),
        ("Last night she eated pizza.", "Last night she ate pizza.", ["Last night she ate pizza."], "eated", "verb_tense", "easy", ["night", "ate", "pizza"]),
        ("He runned as fast as he could.", "He ran as fast as he could.", ["He ran as fast as he could."], "runned", "verb_tense", "easy", ["ran", "fast"]),
        ("They buyed a new tent.", "They bought a new tent.", ["They bought a new tent."], "buyed", "verb_tense", "easy", ["bought", "tent"]),
        ("I seen the rainbow.", "I saw the rainbow.", ["I have seen the rainbow.", "I've seen the rainbow.", "I saw the rainbow."], "seen", "verb_tense", "medium", ["rainbow"]),
        ("She has went home already.", "She has gone home already.", ["She has gone home already.", "She went home already."], "went", "verb_tense", "hard", ["home", "already"]),
        ("We have ate all the grapes.", "We have eaten all the grapes.", ["We have eaten all the grapes.", "We ate all the grapes."], "ate", "verb_tense", "hard", ["grapes"]),
        ("Tomorrow we went to the zoo.", "Tomorrow we will go to the zoo.", ["Tomorrow we are going to the zoo.", "Tomorrow we go to the zoo.", "Tomorrow we'll go to the zoo."], "went", "verb_tense", "medium", ["tomorrow", "zoo"]),
        ("If I was you, I would wait.", "If I were you, I would wait.", ["If I were you, I would wait."], "was", "verb_tense", "hard", ["if", "you", "wait"]),
        ("He didn't went to school.", "He didn't go to school.", ["He did not go to school.", "He didn't go to school."], "went", "verb_tense", "medium", ["school"]),
        ("Did she ate the apple?", "Did she eat the apple?", ["Did she eat the apple?"], "ate", "verb_tense", "medium", ["apple"]),
        ("I am knowing the answer.", "I know the answer.", ["I know the answer."], "am knowing", "verb_tense", "hard", ["know", "answer"]),
        ("She is wanting a snack.", "She wants a snack.", ["She wants a snack.", "She is hungry for a snack."], "is wanting", "verb_tense", "hard", ["snack"]),
        ("When I was little I wear glasses.", "When I was little I wore glasses.", ["When I was little I wore glasses."], "wear", "verb_tense", "medium", ["little", "glasses"]),
        ("By noon they finish the puzzle.", "By noon they had finished the puzzle.", ["By noon they finished the puzzle.", "By noon they had finished the puzzle.", "By noon they will have finished the puzzle."], "finish", "verb_tense", "hard", ["noon", "puzzle"]),
        ("He brang his lunch today.", "He brought his lunch today.", ["He brought his lunch today."], "brang", "verb_tense", "medium", ["lunch", "today"]),
        ("The balloon blowed away.", "The balloon blew away.", ["The balloon blew away."], "blowed", "verb_tense", "easy", ["balloon", "away"]),
        ("I catched the ball.", "I caught the ball.", ["I caught the ball."], "catched", "verb_tense", "easy", ["caught", "ball"]),
        ("She teached us a song.", "She taught us a song.", ["She taught us a song."], "teached", "verb_tense", "easy", ["taught", "song"]),
        ("They swimmed across the pool.", "They swam across the pool.", ["They swam across the pool."], "swimmed", "verb_tense", "easy", ["swam", "pool"]),
        ("He had wrote a letter.", "He had written a letter.", ["He had written a letter.", "He wrote a letter."], "wrote", "verb_tense", "hard", ["letter"]),
        ("The sun rised at six.", "The sun rose at six.", ["The sun rose at six."], "rised", "verb_tense", "medium", ["sun", "six"]),
        ("I drived to the store.", "I drove to the store.", ["I drove to the store."], "drived", "verb_tense", "easy", ["drove", "store"]),
        ("We flied kites on Sunday.", "We flew kites on Sunday.", ["We flew kites on Sunday."], "flied", "verb_tense", "easy", ["flew", "kites", "sunday"]),
        ("She had sang before dinner.", "She had sung before dinner.", ["She had sung before dinner.", "She sang before dinner."], "sang", "verb_tense", "hard", ["dinner"]),
    ]
    for i, (src, gt, acc, bad, cat, diff, meaning) in enumerate(tense, 1):
        items.append(
            _corr(
                f"tense-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Fix verb tense/form while keeping the time meaning.",
                [gt.split()[gt.split().index(w) if w in gt.split() else 0] for w in gt.split()[:3]],
                [bad],
                meaning,
            )
        )

    articles = [
        ("I saw a elephant at the zoo.", "I saw an elephant at the zoo.", ["I saw an elephant at the zoo."], "a elephant", "easy", ["elephant", "zoo"]),
        ("She wants an banana.", "She wants a banana.", ["She wants a banana."], "an banana", "easy", ["banana"]),
        ("He is a honest boy.", "He is an honest boy.", ["He is an honest boy."], "a honest", "hard", ["honest", "boy"]),
        ("Please pass me apple.", "Please pass me an apple.", ["Please pass me an apple.", "Please pass me the apple."], "me apple", "medium", ["apple"]),
        ("We live in United States.", "We live in the United States.", ["We live in the United States."], "in United", "medium", ["united", "states"]),
        ("Moon looks bright tonight.", "The moon looks bright tonight.", ["The moon looks bright tonight."], "Moon looks", "medium", ["moon", "bright"]),
        ("She plays a piano.", "She plays the piano.", ["She plays the piano.", "She plays piano."], "a piano", "hard", ["piano"]),
        ("I need an hour to finish.", "I need an hour to finish.", ["I need an hour to finish."], "", "hard", ["hour", "finish"]),
        ("They adopted a unique kitten.", "They adopted a unique kitten.", ["They adopted a unique kitten."], "", "hard", ["unique", "kitten"]),
        ("He bought a umbrella.", "He bought an umbrella.", ["He bought an umbrella."], "a umbrella", "easy", ["umbrella"]),
        ("Let's sit in a armchair.", "Let's sit in an armchair.", ["Let's sit in an armchair.", "Let us sit in an armchair."], "a armchair", "medium", ["armchair"]),
        ("I go to school by the bus.", "I go to school by bus.", ["I go to school by bus.", "I go to school on the bus."], "by the bus", "hard", ["school", "bus"]),
        ("She is best student in class.", "She is the best student in class.", ["She is the best student in the class.", "She is the best student in class."], "is best", "medium", ["best", "student"]),
        ("We climbed a Alps last year.", "We climbed the Alps last year.", ["We climbed the Alps last year."], "a Alps", "medium", ["alps"]),
        ("Please close a door.", "Please close the door.", ["Please close the door."], "a door", "medium", ["door"]),
        ("I would like a orange.", "I would like an orange.", ["I would like an orange."], "a orange", "easy", ["orange"]),
        ("He works as teacher.", "He works as a teacher.", ["He works as a teacher."], "as teacher", "easy", ["teacher"]),
        ("Sun rises in east.", "The sun rises in the east.", ["The sun rises in the east."], "Sun rises", "medium", ["sun", "east"]),
        ("She has a MBA.", "She has an MBA.", ["She has an MBA."], "a MBA", "hard", ["mba"]),
        ("I found an one-dollar coin.", "I found a one-dollar coin.", ["I found a one-dollar coin."], "an one", "hard", ["coin"]),
    ]
    for i, (src, gt, acc, bad, diff, meaning) in enumerate(articles, 1):
        already = src == gt
        items.append(
            _corr(
                f"art-{i:03d}",
                "already_correct" if already else "articles",
                diff,
                src,
                gt,
                acc,
                "Article choice (a/an/the/zero) must match sound and count.",
                ["preserve meaning"],
                [bad] if bad else [],
                meaning,
            )
        )

    prep = [
        ("She is good in math.", "She is good at math.", ["She is good at math.", "She is good at maths."], "in math", "prepositions", "medium", ["good", "math"]),
        ("We arrived to the station.", "We arrived at the station.", ["We arrived at the station."], "to the station", "prepositions", "medium", ["arrived", "station"]),
        ("He jumped in the pool from the side.", "He jumped into the pool from the side.", ["He jumped into the pool from the side.", "He jumped in the pool from the side."], "", "prepositions", "hard", ["jumped", "pool"]),
        ("The book is in the table.", "The book is on the table.", ["The book is on the table."], "in the table", "prepositions", "easy", ["book", "table"]),
        ("I will see you in Friday.", "I will see you on Friday.", ["I will see you on Friday."], "in Friday", "prepositions", "easy", ["friday"]),
        ("They depend of their coach.", "They depend on their coach.", ["They depend on their coach."], "depend of", "prepositions", "medium", ["depend", "coach"]),
        ("She divided the cake in four pieces.", "She divided the cake into four pieces.", ["She divided the cake into four pieces."], "in four", "prepositions", "medium", ["cake", "four"]),
        ("I am waiting the bus.", "I am waiting for the bus.", ["I am waiting for the bus."], "waiting the", "prepositions", "easy", ["waiting", "bus"]),
        ("Please listen the teacher.", "Please listen to the teacher.", ["Please listen to the teacher."], "listen the", "prepositions", "easy", ["listen", "teacher"]),
        ("He is married with a doctor.", "He is married to a doctor.", ["He is married to a doctor."], "married with", "prepositions", "hard", ["married", "doctor"]),
        ("We discussed about the plan.", "We discussed the plan.", ["We discussed the plan.", "We talked about the plan."], "discussed about", "prepositions", "hard", ["plan"]),
        ("The cat hid under of the bed.", "The cat hid under the bed.", ["The cat hid under the bed."], "under of", "prepositions", "easy", ["cat", "bed"]),
        ("Meet me at noon on the park.", "Meet me at noon in the park.", ["Meet me at noon in the park.", "Meet me at noon at the park."], "on the park", "prepositions", "medium", ["noon", "park"]),
        ("She is interested for science.", "She is interested in science.", ["She is interested in science."], "interested for", "prepositions", "medium", ["interested", "science"]),
        ("I prefer tea than coffee.", "I prefer tea to coffee.", ["I prefer tea to coffee."], "than coffee", "prepositions", "hard", ["tea", "coffee"]),
        ("He accused her for lying.", "He accused her of lying.", ["He accused her of lying."], "accused her for", "prepositions", "hard", ["accused", "lying"]),
        ("They congratulated him for his win.", "They congratulated him on his win.", ["They congratulated him on his win.", "They congratulated him for his win."], "", "prepositions", "hard", ["congratulated", "win"]),
        ("The picture is hanging in the wall.", "The picture is hanging on the wall.", ["The picture is hanging on the wall."], "in the wall", "prepositions", "easy", ["picture", "wall"]),
        ("I looked the window.", "I looked out the window.", ["I looked out the window.", "I looked through the window.", "I looked at the window."], "looked the window", "prepositions", "medium", ["window"]),
        ("She arrived in time for the show.", "She arrived in time for the show.", ["She arrived in time for the show.", "She arrived on time for the show."], "", "already_correct", "hard", ["arrived", "show"]),
    ]
    for i, (src, gt, acc, bad, cat, diff, meaning) in enumerate(prep, 1):
        items.append(
            _corr(
                f"prep-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Choose the natural English preposition (or none).",
                ["natural preposition"],
                [bad] if bad else [],
                meaning,
            )
        )

    spelling = [
        ("The ship is sailing on the see.", "The ship is sailing on the sea.", ["The ship is sailing on the sea."], "see", "spelling", "easy", ["ship", "sailing", "sea"]),
        ("I have free cookies.", "I have three cookies.", ["I have three cookies."], "free", "spelling", "medium", ["cookies"]),
        ("I sink it is fun.", "I think it is fun.", ["I think it is fun."], "sink", "spelling", "easy", ["think", "fun"]),
        ("Their going to the store.", "They're going to the store.", ["They're going to the store.", "They are going to the store."], "Their going", "spelling", "medium", ["going", "store"]),
        ("Its a sunny day.", "It's a sunny day.", ["It's a sunny day.", "It is a sunny day."], "Its a", "spelling", "medium", ["sunny", "day"]),
        ("The dog wagged it's tail.", "The dog wagged its tail.", ["The dog wagged its tail."], "it's tail", "spelling", "hard", ["dog", "tail"]),
        ("Your the best friend I have.", "You're the best friend I have.", ["You're the best friend I have.", "You are the best friend I have."], "Your the", "spelling", "medium", ["best", "friend"]),
        ("I recieve the letter today.", "I receive the letter today.", ["I receive the letter today.", "I received the letter today."], "recieve", "spelling", "easy", ["letter"]),
        ("She is definately coming.", "She is definitely coming.", ["She is definitely coming."], "definately", "spelling", "easy", ["coming"]),
        ("We seperate the colors.", "We separate the colors.", ["We separate the colors.", "We separate the colours."], "seperate", "spelling", "easy", ["colors", "colours"]),
        ("A lot of people came, alot more than we expected.", "A lot of people came, a lot more than we expected.", ["A lot of people came, a lot more than we expected."], "alot", "spelling", "medium", ["people", "expected"]),
        ("He could of helped us.", "He could have helped us.", ["He could have helped us.", "He could've helped us."], "could of", "spelling", "medium", ["helped"]),
        ("Please adress the envelope.", "Please address the envelope.", ["Please address the envelope."], "adress", "spelling", "easy", ["envelope"]),
        ("The whether looks stormy.", "The weather looks stormy.", ["The weather looks stormy."], "whether", "spelling", "medium", ["stormy"]),
        ("Which witch is which?", "Which witch is which?", ["Which witch is which?"], "", "already_correct", "hard", ["witch"]),
        ("I know the no of the house.", "I know the number of the house.", ["I know the number of the house.", "I know the no. of the house."], "the no of", "spelling", "medium", ["house"]),
        ("We except your invitation.", "We accept your invitation.", ["We accept your invitation."], "except", "spelling", "hard", ["invitation"]),
        ("The desert was chocolate cake.", "The dessert was chocolate cake.", ["The dessert was chocolate cake."], "desert was", "spelling", "hard", ["chocolate", "cake"]),
        ("He is quiet a talented singer.", "He is quite a talented singer.", ["He is quite a talented singer."], "quiet a", "spelling", "hard", ["talented", "singer"]),
        ("Lets go to the park.", "Let's go to the park.", ["Let's go to the park.", "Let us go to the park."], "Lets go", "punctuation", "easy", ["park"]),
    ]
    for i, (src, gt, acc, bad, cat, diff, meaning) in enumerate(spelling, 1):
        items.append(
            _corr(
                f"spell-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Fix the homophone/spelling/punctuation error without changing the scene.",
                ["preserve intended word"],
                [bad] if bad else [],
                meaning,
            )
        )

    caps_punct = [
        ("i like pizza.", "I like pizza.", ["I like pizza."], "capitalization", "easy", ["pizza"]),
        ("we visited london last summer.", "We visited London last summer.", ["We visited London last summer."], "capitalization", "easy", ["london", "summer"]),
        ("my friend sam lives on oak street.", "My friend Sam lives on Oak Street.", ["My friend Sam lives on Oak Street."], "capitalization", "medium", ["sam", "oak", "street"]),
        ("she said hello how are you", "She said, \"Hello, how are you?\"", ["She said, \"Hello, how are you?\"", "She said hello, how are you?", "She said, “Hello, how are you?”"], "punctuation", "hard", ["hello"]),
        ("Wait where are you going.", "Wait, where are you going?", ["Wait, where are you going?", "Wait. Where are you going?"], "punctuation", "medium", ["where", "going"]),
        ("Wow that is amazing", "Wow, that is amazing!", ["Wow, that is amazing!", "Wow, that is amazing."], "punctuation", "easy", ["amazing"]),
        ("On monday we have art class.", "On Monday we have art class.", ["On Monday we have art class."], "capitalization", "easy", ["monday", "art"]),
        ("english is fun to learn.", "English is fun to learn.", ["English is fun to learn."], "capitalization", "easy", ["english", "fun"]),
        ("The usa is a large country.", "The USA is a large country.", ["The USA is a large country.", "The U.S.A. is a large country.", "The US is a large country."], "capitalization", "medium", ["large", "country"]),
        ("Please bring pens pencils and paper.", "Please bring pens, pencils, and paper.", ["Please bring pens, pencils, and paper.", "Please bring pens, pencils and paper."], "punctuation", "medium", ["pens", "pencils", "paper"]),
        ("Its time to go isnt it", "It's time to go, isn't it?", ["It's time to go, isn't it?", "It is time to go, is it not?"], "punctuation", "hard", ["time"]),
        ("Dr smith will see you now.", "Dr. Smith will see you now.", ["Dr. Smith will see you now.", "Dr Smith will see you now."], "capitalization", "medium", ["smith"]),
        ("i cant find my keys.", "I can't find my keys.", ["I can't find my keys.", "I cannot find my keys."], "punctuation", "easy", ["keys"]),
        ("What a beautiful day we are having.", "What a beautiful day we are having!", ["What a beautiful day we are having!", "What a beautiful day we are having."], "punctuation", "medium", ["beautiful", "day"]),
        ("however we decided to stay.", "However, we decided to stay.", ["However, we decided to stay."], "punctuation", "medium", ["decided", "stay"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(caps_punct, 1):
        items.append(
            _corr(
                f"cap-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Fix capitalization and/or punctuation only as needed.",
                ["readable sentence"],
                [],
                meaning,
            )
        )

    pronouns = [
        ("Me and him went to the store.", "He and I went to the store.", ["He and I went to the store.", "I went to the store with him."], "Me and him", "pronouns", "medium", ["store"]),
        ("Give the book to I.", "Give the book to me.", ["Give the book to me."], "to I", "pronouns", "easy", ["book"]),
        ("This is mines.", "This is mine.", ["This is mine."], "mines", "pronouns", "easy", ["mine"]),
        ("Everyone should bring their pencil.", "Everyone should bring their pencil.", ["Everyone should bring their pencil.", "Everyone should bring his or her pencil."], "", "already_correct", "hard", ["everyone", "pencil"]),
        ("The team won its first game.", "The team won its first game.", ["The team won its first game."], "", "already_correct", "medium", ["team", "game"]),
        ("Her and I are friends.", "She and I are friends.", ["She and I are friends."], "Her and I", "pronouns", "medium", ["friends"]),
        ("Between you and I, the secret is safe.", "Between you and me, the secret is safe.", ["Between you and me, the secret is safe."], "you and I", "pronouns", "hard", ["secret", "safe"]),
        ("Who did you give it to?", "Who did you give it to?", ["Who did you give it to?", "Whom did you give it to?"], "", "already_correct", "hard", ["give"]),
        ("The dog hurt hisself.", "The dog hurt himself.", ["The dog hurt himself."], "hisself", "pronouns", "easy", ["dog"]),
        ("They invited my sister and I.", "They invited my sister and me.", ["They invited my sister and me."], "and I", "pronouns", "hard", ["sister", "invited"]),
        ("It was her who called.", "It was she who called.", ["It was she who called.", "She was the one who called.", "It was her who called."], "", "pronouns", "hard", ["called"]),
        ("Each of the girls brought her lunch.", "Each of the girls brought her lunch.", ["Each of the girls brought her lunch.", "Each of the girls brought their lunch."], "", "already_correct", "hard", ["girls", "lunch"]),
        ("Somebody left their backpack.", "Somebody left their backpack.", ["Somebody left their backpack.", "Somebody left his or her backpack."], "", "already_correct", "medium", ["backpack"]),
        ("Us kids want to play outside.", "We kids want to play outside.", ["We kids want to play outside.", "We want to play outside."], "Us kids", "pronouns", "medium", ["play", "outside"]),
        ("The prize belongs to she.", "The prize belongs to her.", ["The prize belongs to her."], "to she", "pronouns", "easy", ["prize"]),
    ]
    for i, (src, gt, acc, bad, cat, diff, meaning) in enumerate(pronouns, 1):
        items.append(
            _corr(
                f"pro-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Use the case-appropriate pronoun; accept established singular they.",
                ["pronoun case"],
                [bad] if bad else [],
                meaning,
            )
        )

    word_order = [
        ("Always she is late.", "She is always late.", ["She is always late."], "word_order", "easy", ["always", "late"]),
        ("I yesterday went home.", "I went home yesterday.", ["Yesterday I went home.", "I went home yesterday."], "word_order", "easy", ["home", "yesterday"]),
        ("Never I have seen a comet.", "Never have I seen a comet.", ["I have never seen a comet.", "Never have I seen a comet."], "word_order", "hard", ["comet"]),
        ("He speaks well English.", "He speaks English well.", ["He speaks English well."], "word_order", "medium", ["english", "well"]),
        ("We will at noon eat.", "We will eat at noon.", ["We will eat at noon."], "word_order", "easy", ["eat", "noon"]),
        ("Beautiful a garden they have.", "They have a beautiful garden.", ["They have a beautiful garden."], "word_order", "medium", ["beautiful", "garden"]),
        ("Why you are crying?", "Why are you crying?", ["Why are you crying?"], "word_order", "easy", ["crying"]),
        ("Where he did go?", "Where did he go?", ["Where did he go?"], "word_order", "easy", ["go"]),
        ("Can you tell me where is the library?", "Can you tell me where the library is?", ["Can you tell me where the library is?"], "word_order", "hard", ["library"]),
        ("Only can she play the flute.", "Only she can play the flute.", ["Only she can play the flute.", "She can only play the flute."], "word_order", "hard", ["flute"]),
        ("I have a red big balloon.", "I have a big red balloon.", ["I have a big red balloon."], "word_order", "medium", ["balloon", "red"]),
        ("She made for me a cake.", "She made a cake for me.", ["She made a cake for me.", "She made me a cake."], "word_order", "medium", ["cake"]),
        ("Off took he his hat.", "He took off his hat.", ["He took off his hat.", "He took his hat off."], "word_order", "medium", ["hat"]),
        ("Rarely they eat dessert.", "Rarely do they eat dessert.", ["They rarely eat dessert.", "Rarely do they eat dessert."], "word_order", "hard", ["dessert"]),
        ("The kids played happily in the yard all afternoon.", "The kids played happily in the yard all afternoon.", ["The kids played happily in the yard all afternoon."], "already_correct", "easy", ["kids", "yard"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(word_order, 1):
        items.append(
            _corr(
                f"ord-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Restore natural English word order.",
                ["natural order"],
                [],
                meaning,
            )
        )

    fragments_runons = [
        ("Because I was tired.", "I sat down because I was tired.", ["Because I was tired, I sat down.", "I was tired.", "I sat down because I was tired."], "fragments", "medium", ["tired"]),
        ("Running down the hill.", "The kids were running down the hill.", ["Running down the hill, the kids laughed.", "The kids were running down the hill."], "fragments", "medium", ["hill"]),
        ("When the bell rang", "When the bell rang, we stood up.", ["When the bell rang, we stood up.", "The bell rang."], "fragments", "medium", ["bell"]),
        ("I love ice cream I eat it every day.", "I love ice cream. I eat it every day.", ["I love ice cream. I eat it every day.", "I love ice cream, and I eat it every day.", "I love ice cream; I eat it every day."], "run_on", "easy", ["ice", "cream", "every"]),
        ("The movie was long we still enjoyed it.", "The movie was long, but we still enjoyed it.", ["The movie was long, but we still enjoyed it.", "The movie was long. We still enjoyed it."], "run_on", "easy", ["movie", "enjoyed"]),
        ("She packed her bag she forgot her lunch.", "She packed her bag, but she forgot her lunch.", ["She packed her bag, but she forgot her lunch.", "She packed her bag. She forgot her lunch."], "run_on", "medium", ["bag", "lunch"]),
        ("It rained all night the streets were wet.", "It rained all night, so the streets were wet.", ["It rained all night, so the streets were wet.", "It rained all night. The streets were wet."], "run_on", "medium", ["rained", "streets"]),
        ("Although the test was hard. I finished it.", "Although the test was hard, I finished it.", ["Although the test was hard, I finished it."], "fragments", "medium", ["test", "finished"]),
        ("My favorite color is blue it reminds me of the sky.", "My favorite color is blue; it reminds me of the sky.", ["My favorite color is blue. It reminds me of the sky.", "My favorite color is blue because it reminds me of the sky.", "My favorite color is blue; it reminds me of the sky."], "run_on", "medium", ["blue", "sky"]),
        ("Stop. Wait. Listen.", "Stop, wait, and listen.", ["Stop, wait, and listen.", "Stop. Wait. Listen."], "sentence_structure", "hard", ["stop", "wait", "listen"]),
        ("The boy who lives next door.", "The boy who lives next door is kind.", ["The boy who lives next door is kind.", "I know the boy who lives next door."], "fragments", "medium", ["boy", "door"]),
        ("He was hungry he made a sandwich he sat down.", "He was hungry, so he made a sandwich and sat down.", ["He was hungry. He made a sandwich and sat down.", "He was hungry, so he made a sandwich and sat down."], "run_on", "hard", ["hungry", "sandwich"]),
        ("If you finish your homework", "If you finish your homework, you may play.", ["If you finish your homework, you may play.", "Finish your homework."], "fragments", "medium", ["homework"]),
        ("I wanted to go however I had to stay home.", "I wanted to go; however, I had to stay home.", ["I wanted to go; however, I had to stay home.", "I wanted to go. However, I had to stay home.", "I wanted to go, but I had to stay home."], "run_on", "hard", ["wanted", "stay", "home"]),
        ("Which made everyone laugh.", "The joke made everyone laugh.", ["The joke made everyone laugh.", "That made everyone laugh."], "fragments", "hard", ["laugh"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(fragments_runons, 1):
        items.append(
            _corr(
                f"str-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Repair fragments or run-ons while keeping the same ideas.",
                ["complete grammatical sentence(s)"],
                [],
                meaning,
            )
        )

    awkward = [
        ("The reason is because I was late.", "The reason is that I was late.", ["The reason is that I was late.", "I was late.", "I was late; that is the reason."], "awkward_phrasing", "hard", ["late"]),
        ("Me hungry now.", "I am hungry now.", ["I am hungry now.", "I'm hungry now."], "awkward_phrasing", "easy", ["hungry"]),
        ("This thing it is broken.", "This thing is broken.", ["This thing is broken.", "It is broken."], "awkward_phrasing", "easy", ["broken"]),
        ("In my opinion, I think we should wait.", "I think we should wait.", ["I think we should wait.", "In my opinion, we should wait."], "redundant_wording", "medium", ["wait"]),
        ("She returned back to her seat.", "She returned to her seat.", ["She returned to her seat.", "She went back to her seat."], "redundant_wording", "easy", ["seat"]),
        ("He is a person who is kind.", "He is kind.", ["He is kind.", "He is a kind person."], "concise_rewriting", "medium", ["kind"]),
        ("At this point in time we need help.", "We need help now.", ["We need help now.", "At this time we need help.", "We need help at this time."], "concise_rewriting", "medium", ["help"]),
        ("The end result was a tie.", "The result was a tie.", ["The result was a tie.", "It ended in a tie."], "redundant_wording", "medium", ["tie"]),
        ("Please repeat again the question.", "Please repeat the question.", ["Please repeat the question.", "Please say the question again."], "redundant_wording", "easy", ["question"]),
        ("I myself personally disagree.", "I disagree.", ["I disagree.", "I personally disagree."], "redundant_wording", "medium", ["disagree"]),
        ("It is necessary that you must sit.", "You must sit.", ["You must sit.", "It is necessary that you sit."], "concise_rewriting", "medium", ["sit"]),
        ("The actual fact is we won.", "The fact is we won.", ["The fact is we won.", "We won."], "redundant_wording", "medium", ["won"]),
        ("Close proximity to the school helps.", "Proximity to the school helps.", ["Being close to the school helps.", "Proximity to the school helps.", "Living near the school helps."], "redundant_wording", "hard", ["school"]),
        ("She whispered quietly to me.", "She whispered to me.", ["She whispered to me.", "She quietly whispered to me."], "redundant_wording", "hard", ["whispered"]),
        ("Due to the fact that it rained, we stayed in.", "Because it rained, we stayed in.", ["Because it rained, we stayed in.", "Due to the rain, we stayed in."], "concise_rewriting", "medium", ["rained", "stayed"]),
        ("I will try and finish the work.", "I will try to finish the work.", ["I will try to finish the work."], "natural_english", "medium", ["finish", "work"]),
        ("Can you borrow me a pencil?", "Can you lend me a pencil?", ["Can you lend me a pencil?", "May I borrow a pencil?"], "word_choice", "medium", ["pencil"]),
        ("I am interesting in dinosaurs.", "I am interested in dinosaurs.", ["I am interested in dinosaurs."], "word_choice", "easy", ["dinosaurs"]),
        ("The food was very delicious tasty.", "The food was delicious.", ["The food was delicious.", "The food was very tasty."], "redundant_wording", "easy", ["food"]),
        ("He did a mistake on the test.", "He made a mistake on the test.", ["He made a mistake on the test."], "word_choice", "medium", ["mistake", "test"]),
        ("I have 10 years.", "I am 10 years old.", ["I am 10 years old.", "I am ten years old."], "natural_english", "easy", ["years"]),
        ("Open the light, please.", "Turn on the light, please.", ["Turn on the light, please.", "Please turn on the light."], "natural_english", "medium", ["light"]),
        ("She explained me the rule.", "She explained the rule to me.", ["She explained the rule to me."], "natural_english", "medium", ["rule"]),
        ("I very like this song.", "I like this song very much.", ["I like this song very much.", "I really like this song."], "natural_english", "easy", ["song"]),
        ("How do you call this in English?", "What do you call this in English?", ["What do you call this in English?", "What is this called in English?"], "natural_english", "medium", ["english"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(awkward, 1):
        items.append(
            _corr(
                f"awk-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Make the English natural and concise without changing the claim.",
                ["natural idiomatic English"],
                [],
                meaning,
            )
        )

    formal = [
        ("Wanna go to the park?", "Would you like to go to the park?", ["Would you like to go to the park?", "Do you want to go to the park?"], "formal_informal", "easy", ["park"]),
        ("Kids gotta finish homework first.", "Children have to finish their homework first.", ["Children have to finish their homework first.", "Kids have to finish homework first."], "formal_informal", "medium", ["homework"]),
        ("Yeah it was kinda fun.", "Yes, it was rather fun.", ["Yes, it was rather fun.", "Yes, it was kind of fun.", "Yes, it was quite fun."], "formal_informal", "medium", ["fun"]),
        ("Please refrain from running in the hallways.", "Please refrain from running in the hallways.", ["Please refrain from running in the hallways."], "already_correct", "easy", ["running", "hallways"]),
        ("I ain't got no pencil.", "I do not have a pencil.", ["I do not have a pencil.", "I don't have a pencil.", "I have no pencil."], "formal_informal", "easy", ["pencil"]),
        ("Could you possibly assist me with this problem?", "Could you possibly assist me with this problem?", ["Could you possibly assist me with this problem?", "Could you help me with this problem?"], "already_correct", "medium", ["assist", "problem"]),
        ("Gimme that book.", "Please give me that book.", ["Please give me that book.", "Give me that book, please."], "formal_informal", "easy", ["book"]),
        ("We should of told them sooner.", "We should have told them sooner.", ["We should have told them sooner."], "formal_informal", "medium", ["told", "sooner"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(formal, 1):
        items.append(
            _corr(
                f"reg-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Register: keep meaning; prefer standard school English.",
                ["standard English"],
                [],
                meaning,
            )
        )

    multi = [
        ("she dont has no apples", "She doesn't have any apples.", ["She doesn't have any apples.", "She does not have any apples.", "She has no apples."], "multiple_errors", "hard", ["apples"]),
        ("Yesterday we was goed to see sea.", "Yesterday we went to see the sea.", ["Yesterday we went to see the sea."], "multiple_errors", "adversarial", ["yesterday", "sea"]),
        ("Me and him don't likes this free cookies.", "He and I don't like these three cookies.", ["He and I don't like these three cookies.", "He and I do not like these three cookies."], "multiple_errors", "adversarial", ["cookies"]),
        ("Its raining and we was forgot our umbrella's.", "It's raining and we forgot our umbrellas.", ["It's raining and we forgot our umbrellas.", "It is raining, and we forgot our umbrellas."], "multiple_errors", "hard", ["raining", "umbrellas"]),
        ("He have went to london last monday.", "He went to London last Monday.", ["He went to London last Monday.", "He has gone to London."], "multiple_errors", "hard", ["london", "monday"]),
        ("Why you didn't told I the answer.", "Why didn't you tell me the answer?", ["Why didn't you tell me the answer?", "Why did you not tell me the answer?"], "multiple_errors", "hard", ["answer"]),
        ("Them kids is playing noisy in indoor.", "Those kids are playing noisily indoors.", ["Those kids are playing noisily indoors.", "The kids are playing noisily indoors."], "multiple_errors", "adversarial", ["kids", "playing"]),
        ("I seen she at the library she don't wave.", "I saw her at the library; she didn't wave.", ["I saw her at the library, but she didn't wave.", "I saw her at the library; she didn't wave."], "multiple_errors", "adversarial", ["library", "wave"]),
        ("There is many reason why this don't work.", "There are many reasons why this doesn't work.", ["There are many reasons why this doesn't work.", "There are many reasons why this does not work."], "multiple_errors", "hard", ["reasons", "work"]),
        ("A apple was eated by he yesterday.", "An apple was eaten by him yesterday.", ["He ate an apple yesterday.", "An apple was eaten by him yesterday."], "multiple_errors", "hard", ["apple", "yesterday"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(multi, 1):
        items.append(
            _corr(
                f"mul-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Fix every error; keep the same facts.",
                ["all grammar errors fixed", "same meaning"],
                [],
                meaning,
            )
        )

    meaning_critical = [
        ("Only I ate the cookies.", "Only I ate the cookies.", ["Only I ate the cookies."], "already_correct", "hard", ["only", "ate", "cookies"], "Do not move only; it changes who ate."),
        ("I only ate the cookies.", "I only ate the cookies.", ["I only ate the cookies."], "already_correct", "hard", ["only", "ate", "cookies"], "Do not move only."),
        ("The teacher said the student was wrong.", "The teacher said the student was wrong.", ["The teacher said the student was wrong."], "already_correct", "medium", ["teacher", "student", "wrong"], "Do not swap who was wrong."),
        ("She almost failed every test.", "She almost failed every test.", ["She almost failed every test."], "already_correct", "adversarial", ["almost", "failed", "every", "test"], "almost failed every ≠ failed almost every."),
        ("She failed almost every test.", "She failed almost every test.", ["She failed almost every test."], "already_correct", "adversarial", ["failed", "almost", "every", "test"], "Preserve adverb placement."),
        ("I didn't steal the cake because I was hungry.", "I didn't steal the cake because I was hungry.", ["I didn't steal the cake because I was hungry."], "already_correct", "adversarial", ["steal", "cake", "hungry"], "Negation scope is meaningful."),
        ("Visiting relatives can be a nuisance.", "Visiting relatives can be a nuisance.", ["Visiting relatives can be a nuisance."], "ambiguity", "adversarial", ["visiting", "relatives"], "Do not pick one reading unless asked; keep the original wording if grammatical."),
        ("The chicken is ready to eat.", "The chicken is ready to eat.", ["The chicken is ready to eat."], "ambiguity", "adversarial", ["chicken", "ready", "eat"], "Keep the original grammatical sentence."),
        ("We need more intelligent students.", "We need more intelligent students.", ["We need more intelligent students."], "already_correct", "hard", ["intelligent", "students"], "Do not change more intelligent vs more students."),
        ("Stop clubbing baby seals.", "Stop clubbing baby seals.", ["Stop clubbing baby seals."], "adversarial", "adversarial", ["stop", "clubbing"], "Do not 'fix' a grammatical if shocking sentence by rewriting topic; keep syntax. Kid-eval: treat as grammar-only identity."),
        ("Flying planes can be dangerous.", "Flying planes can be dangerous.", ["Flying planes can be dangerous."], "ambiguity", "adversarial", ["flying", "planes"], "Grammatical as-is."),
        ("He promised me to help.", "He promised to help me.", ["He promised to help me.", "He promised me that he would help."], "meaning_preservation", "hard", ["promised", "help"]),
        ("All that glitters are not gold.", "All that glitters is not gold.", ["All that glitters is not gold."], "subject_verb_agreement", "hard", ["glitters", "gold"]),
        ("None of the pie were eaten.", "None of the pie was eaten.", ["None of the pie was eaten.", "None of the pie were eaten."], "subject_verb_agreement", "hard", ["pie", "eaten"]),
        ("The data is on the table.", "The data is on the table.", ["The data is on the table.", "The data are on the table."], "multiple_valid", "hard", ["data", "table"]),
        ("Color the gray square grey.", "Color the gray square grey.", ["Color the gray square grey.", "Colour the grey square grey.", "Color the gray square gray."], "multiple_valid", "hard", ["square"]),
        ("Towards evening, the sky grew toward pink.", "Toward evening, the sky grew toward pink.", ["Toward evening, the sky grew toward pink.", "Towards evening, the sky grew towards pink.", "Toward evening, the sky grew pink."], "multiple_valid", "adversarial", ["evening", "sky"]),
        ("If I would have known, I would have waited.", "If I had known, I would have waited.", ["If I had known, I would have waited.", "Had I known, I would have waited."], "subtle_distinctions", "hard", ["known", "waited"]),
        ("I could care less about the score.", "I couldn't care less about the score.", ["I couldn't care less about the score.", "I could not care less about the score."], "subtle_distinctions", "adversarial", ["care", "score"]),
        ("There's lots of reasons to smile.", "There are lots of reasons to smile.", ["There are lots of reasons to smile.", "There are a lot of reasons to smile."], "subject_verb_agreement", "medium", ["reasons", "smile"]),
        ("Less people came than we expected.", "Fewer people came than we expected.", ["Fewer people came than we expected.", "Less people came than we expected."], "subtle_distinctions", "hard", ["people", "expected"]),
        ("She is taller then her brother.", "She is taller than her brother.", ["She is taller than her brother."], "word_choice", "easy", ["taller", "brother"]),
        ("Who's backpack is this?", "Whose backpack is this?", ["Whose backpack is this?"], "spelling", "medium", ["backpack"]),
        ("Your late again, and your homework is missing.", "You're late again, and your homework is missing.", ["You're late again, and your homework is missing.", "You are late again, and your homework is missing."], "context_dependent", "hard", ["late", "homework"]),
        ("The man who the dog bit was kind.", "The man whom the dog bit was kind.", ["The man whom the dog bit was kind.", "The man who the dog bit was kind.", "The man the dog bit was kind."], "multiple_valid", "hard", ["man", "dog", "kind"]),
        ("I feel badly about the mistake.", "I feel bad about the mistake.", ["I feel bad about the mistake.", "I feel badly about the mistake."], "multiple_valid", "hard", ["mistake"]),
        ("Go slow.", "Go slowly.", ["Go slowly.", "Go slow."], "multiple_valid", "medium", ["go"]),
        ("Neither the teacher nor the students was ready.", "Neither the teacher nor the students were ready.", ["Neither the teacher nor the students were ready."], "subject_verb_agreement", "hard", ["teacher", "students", "ready"]),
        ("Each of the boxes contain a toy.", "Each of the boxes contains a toy.", ["Each of the boxes contains a toy."], "subject_verb_agreement", "medium", ["boxes", "toy"]),
        ("Here comes the children.", "Here come the children.", ["Here come the children."], "subject_verb_agreement", "medium", ["children"]),
    ]
    for i, row in enumerate(meaning_critical, 1):
        if len(row) == 7:
            src, gt, acc, cat, diff, meaning, expl = row
        else:
            src, gt, acc, cat, diff, meaning = row
            expl = "Preserve meaning exactly; grammar-only edits."
        items.append(
            _corr(
                f"sem-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                expl,
                ["preserve meaning", "do not over-edit"],
                [],
                meaning,
            )
        )

    already = [
        ("The quick brown fox jumps over the lazy dog.", "easy"),
        ("She doesn't like apples.", "easy"),
        ("Yesterday I went to the park.", "easy"),
        ("He has two cats.", "easy"),
        ("The ship is sailing on the sea.", "easy"),
        ("I think it is fun.", "easy"),
        ("We were playing tag.", "easy"),
        ("Please sit down and listen.", "easy"),
        ("My grandmother bakes the best bread.", "easy"),
        ("After dinner, we washed the dishes.", "medium"),
        ("Neither option looks appealing.", "medium"),
        ("The books on the shelf belong to Maya.", "medium"),
        ("Whoever finishes first may choose a prize.", "hard"),
        ("The committee has reached a decision.", "hard"),
        ("Rarely have we seen such kindness.", "hard"),
        ("She asked whether we could stay.", "medium"),
        ("Bring a coat in case it rains.", "easy"),
        ("Those were the days we remember fondly.", "medium"),
        ("A herd of elephants was crossing the river.", "hard"),
        ("Reading quietly, the child smiled.", "medium"),
        ("It is I who am responsible.", "adversarial"),
        ("Had I known, I would have called.", "hard"),
        ("The more you practice, the better you get.", "easy"),
        ("No one but them saw the comet.", "hard"),
        ("Coffee, tea, or juice will be served.", "easy"),
    ]
    for i, (sent, diff) in enumerate(already, 1):
        tokens = [w.strip(".,!?;:").lower() for w in sent.split() if w.strip(".,!?;:")]
        items.append(
            _corr(
                f"ok-{i:03d}",
                "already_correct",
                diff,
                sent,
                sent,
                [sent],
                "Already grammatical: repeat unchanged; do not invent problems.",
                ["identity", "no unnecessary rewrite"],
                [],
                tokens[:6],
            )
        )

    # Kid-tutor mirrors from SuperApp
    tutor = [
        ("She don't like apples.", "She doesn't like apples.", ["She does not like apples."], "subject_verb_agreement"),
        ("Yesterday I goed to the park.", "Yesterday I went to the park.", [], "verb_tense"),
        ("He have two cats.", "He has two cats.", [], "subject_verb_agreement"),
        ("The ship is sailing on the see.", "The ship is sailing on the sea.", [], "spelling"),
        ("I have free cookies.", "I have three cookies.", [], "spelling"),
        ("We was playing tag.", "We were playing tag.", [], "subject_verb_agreement"),
        ("I sink it is fun.", "I think it is fun.", [], "spelling"),
        ("They was hungry after practice.", "They were hungry after practice.", [], "subject_verb_agreement"),
        ("She have a blue backpack.", "She has a blue backpack.", [], "subject_verb_agreement"),
        ("I buyed milk at the store.", "I bought milk at the store.", [], "verb_tense"),
    ]
    for i, (src, gt, acc, cat) in enumerate(tutor, 1):
        items.append(
            _corr(
                f"tutor-{i:03d}",
                cat,
                "easy",
                src,
                gt,
                acc,
                "Core SuperApp catch-the-mistake pattern.",
                ["fix the single kid-level error"],
                [],
                [w.strip(".").lower() for w in gt.split()[:5]],
            )
        )

    # Judges
    complete_cases = [
        ("The cat sat on the", "mat", "mat", True, "easy", "Valid noun ending"),
        ("The cat sat on the", "soft red cushion", "mat", True, "medium", "Alternate valid ending"),
        ("The cat sat on the", "because", "mat", False, "easy", "Does not finish the thought"),
        ("The cat sat on the", "asdfgh", "mat", False, "easy", "Nonsense"),
        ("The cat sat on the", "", "mat", False, "easy", "Empty"),
        ("I like to eat", "apples", "apples", True, "easy", "Food ending"),
        ("I like to eat", "soccer", "apples", False, "medium", "Does not fit eat"),
        ("I like to eat", "pizza with extra cheese", "apples", True, "medium", "Longer valid food"),
        ("At night I look at the", "stars", "moon", True, "easy", "Alternate sky object"),
        ("At night I look at the", "refrigerator", "moon", False, "medium", "Unrelated"),
        ("My favorite color is", "blue", "green", True, "easy", "Any color"),
        ("My favorite color is", "running", "green", False, "easy", "Not a color"),
        ("Please close the", "door", "window", True, "easy", "Closeable object"),
        ("Please close the", "happiness", "window", False, "hard", "Abstract noun mismatch"),
        ("We planted a", "tree", "flower", True, "easy", "Plantable"),
        ("We planted a", "bicycle", "flower", False, "medium", "Not plantable"),
        ("The baby drank", "milk", "water", True, "easy", "Drink"),
        ("The baby drank", "a mountain", "water", False, "easy", "Impossible"),
        ("Grandpa read me a", "story", "book", True, "easy", "Readable"),
        ("Grandpa read me a", "truck", "book", False, "medium", "Not readable"),
        ("After school I play", "soccer", "tag", True, "easy", "Game/sport"),
        ("After school I play", "invisible thunder soup", "tag", False, "hard", "Nonsense phrase"),
        ("The bird built a", "nest", "house", True, "easy", "Bird home"),
        ("The bird built a", "skyscraper of pudding", "house", False, "hard", "Silly/surreal"),
        ("She put on her", "coat", "shoes", True, "easy", "Clothing"),
        ("She put on her", "Tuesday", "shoes", False, "medium", "Not wearable"),
        ("I washed my", "hands", "face", True, "easy", "Body part"),
        ("I washed my", "planet", "face", False, "medium", "Odd"),
        ("The baker made a", "cake", "pie", True, "easy", "Baked good"),
        ("The baker made a", "spaceship", "pie", False, "medium", "Not bakery"),
        ("We waited for the", "bus", "train", True, "easy", "Vehicle"),
        ("We waited for the", "blurble", "train", False, "easy", "Nonce word"),
        ("Mom packed my", "lunch", "backpack", True, "easy", "Packable"),
        ("Mom packed my", "because why", "backpack", False, "easy", "Fragment"),
        ("The dog chased the", "ball", "cat", True, "easy", "Chaseable"),
        ("The dog chased the", "quietly", "cat", False, "medium", "Adverb not noun"),
        ("I drew a picture of a", "dragon", "house", True, "easy", "Drawable noun"),
        ("I drew a picture of a", "and", "house", False, "easy", "Conjunction"),
        ("Please pass the", "salt", "butter", True, "easy", "Table item"),
        ("Please pass the", "galaxy", "butter", False, "hard", "Not a table item"),
        ("He climbed the", "tree", "ladder", True, "easy", "Climbable"),
        ("He climbed the", "song", "ladder", False, "medium", "Not climbable"),
        ("The kids splashed in the", "puddle", "pool", True, "easy", "Water"),
        ("The kids splashed in the", "homework", "pool", False, "medium", "Mismatch"),
        ("I hear a", "bird", "car", True, "easy", "Sound source"),
        ("I hear a", "blanket", "car", False, "hard", "Weak sound source"),
        ("She opened the", "gift", "door", True, "easy", "Openable"),
        ("She opened the", "sleep", "door", False, "medium", "Not openable"),
        ("We blew out the", "candles", "candle", True, "easy", "Birthday schema"),
        ("We blew out the", "mountain", "candle", False, "easy", "Mismatch"),
        ("The cat sat on the", "idk", "mat", False, "medium", "Give-up is not a completion"),
        ("The frog sat on a", "lily pad", "log", True, "medium", "Multiword valid"),
        ("The frog sat on a", "lily", "log", True, "medium", "Partial but possible"),
        ("Dad parked the", "car", "truck", True, "easy", "Vehicle"),
        ("Dad parked the", "ocean", "truck", False, "easy", "Cannot park ocean"),
        ("I brushed my", "teeth", "hair", True, "easy", "Hygiene"),
        ("I brushed my", "clouds", "hair", False, "easy", "Nonsense"),
        ("The snow fell on the", "ground", "roof", True, "easy", "Surface"),
        ("The snow fell on the", "yesterday", "roof", False, "medium", "Time word"),
        ("Our class walked to the", "library", "gym", True, "easy", "School place"),
        ("Our class walked to the", "because lunch", "gym", False, "easy", "Broken"),
    ]
    for i, row in enumerate(complete_cases, 1):
        stem, child, sample, expected, diff, expl = row
        items.append(
            _judge_complete(
                f"jc-{i:03d}",
                diff,
                stem,
                child,
                sample,
                expected,
                expl,
                ["json", "correct boolean", "kid-safe real English"],
            )
        )

    mistake_cases = [
        ("She don't like apples.", "She doesn't like apples.", "She doesn't like apples.", True, "easy", "Canonical fix"),
        ("She don't like apples.", "She does not like apples.", "She doesn't like apples.", True, "medium", "Equivalent paraphrase"),
        ("She don't like apples.", "She don't like apples.", "She doesn't like apples.", False, "easy", "Repeated error"),
        ("She don't like apples.", "I like bananas.", "She doesn't like apples.", False, "easy", "Different sentence"),
        ("Yesterday I goed to the park.", "Yesterday I went to the park.", "Yesterday I went to the park.", True, "easy", "goed→went"),
        ("Yesterday I goed to the park.", "Yesterday I goed to the park.", "Yesterday I went to the park.", False, "easy", "Unfixed"),
        ("Yesterday I goed to the park.", "I went yesterday to the park.", "Yesterday I went to the park.", True, "hard", "Same error fixed, word order ok"),
        ("He have two cats.", "He has two cats.", "He has two cats.", True, "easy", "has"),
        ("He have two cats.", "He have two cats.", "He has two cats.", False, "easy", "Unfixed"),
        ("He have two cats.", "He has two dogs.", "He has two cats.", False, "hard", "Changed meaning"),
        ("The ship is sailing on the see.", "The ship is sailing on the sea.", "The ship is sailing on the sea.", True, "easy", "sea"),
        ("The ship is sailing on the see.", "The ship is sailing on the see.", "The ship is sailing on the sea.", False, "easy", "Unfixed homophone"),
        ("I have free cookies.", "I have three cookies.", "I have three cookies.", True, "medium", "three"),
        ("I have free cookies.", "I have free cookies.", "I have three cookies.", False, "medium", "Unfixed"),
        ("I have free cookies.", "The cookies are free.", "I have three cookies.", False, "hard", "Wrong reading of free"),
        ("We was playing tag.", "We were playing tag.", "We were playing tag.", True, "easy", "were"),
        ("We was playing tag.", "We was playing tag.", "We were playing tag.", False, "easy", "Unfixed"),
        ("We was playing tag.", "They were playing soccer.", "We were playing tag.", False, "medium", "Wrong scene"),
        ("I sink it is fun.", "I think it is fun.", "I think it is fun.", True, "easy", "think"),
        ("I sink it is fun.", "I sink it is fun.", "I think it is fun.", False, "easy", "Unfixed"),
        ("She don't like apples.", "She doesn't likes apples.", "She doesn't like apples.", False, "hard", "New error introduced"),
        ("He have two cats.", "He's got two cats.", "He has two cats.", True, "hard", "Informal but valid fix"),
        ("We was playing tag.", "We were playing a game of tag.", "We were playing tag.", True, "medium", "Slight expansion ok"),
        ("Yesterday I goed to the park.", "Yesterday I went home.", "Yesterday I went to the park.", False, "medium", "Lost park"),
        ("The ship is sailing on the see.", "The ship sails on the sea.", "The ship is sailing on the sea.", True, "hard", "Tense tweak still fixes homophone"),
        ("I sink it is fun.", "I think that's fun.", "I think it is fun.", True, "hard", "Paraphrase"),
        ("She don't like apples.", "Apples aren't liked by she.", "She doesn't like apples.", False, "adversarial", "Broken passive"),
        ("He have two cats.", "Two cats he has.", "He has two cats.", True, "adversarial", "Awkward but agrees"),
        ("We was playing tag.", "Tag was being played by we.", "We were playing tag.", False, "adversarial", "Wrong pronoun case"),
        ("I have free cookies.", "I have 3 cookies.", "I have three cookies.", True, "medium", "Numeral ok"),
        ("She don't like apples.", "She do not like apples.", "She doesn't like apples.", False, "hard", "Still disagrees"),
        ("Yesterday I goed to the park.", "Yesterday I had gone to the park.", "Yesterday I went to the park.", True, "hard", "Past perfect still past-of-go"),
        ("He have two cats.", "He owns two cats.", "He has two cats.", True, "hard", "Owns preserves meaning"),
        ("The ship is sailing on the see.", "The boat is on the sea.", "The ship is sailing on the sea.", False, "hard", "Dropped sailing; maybe too different"),
        ("I sink it is fun.", "Thinking is fun.", "I think it is fun.", False, "medium", "Different proposition"),
        ("We was playing tag.", "We are playing tag.", "We were playing tag.", False, "medium", "Wrong tense vs original past"),
        ("She don't like apples.", "She never liked apples.", "She doesn't like apples.", True, "adversarial", "Stronger but same dislike"),
        ("He have two cats.", "He have got two cats.", "He has two cats.", False, "medium", "Error remains"),
        ("I have free cookies.", "I have tree cookies.", "I have three cookies.", False, "hard", "New homophone error"),
        ("The ship is sailing on the see.", "The ship is sailing on the ocean.", "The ship is sailing on the sea.", True, "hard", "ocean ≈ sea"),
        ("Yesterday I goed to the park.", "I goed yesterday to park.", "Yesterday I went to the park.", False, "easy", "Error remains"),
        ("We was playing tag.", "We were playing.", "We were playing tag.", False, "hard", "Dropped tag"),
        ("She don't like apples.", "She doesn't like apple.", "She doesn't like apples.", True, "hard", "Singular apple still ok-ish? mark true as equivalent food dislike - actually number change. Set false."),
        ("He have two cats.", "He has a pair of cats.", "He has two cats.", True, "medium", "pair of cats = two"),
        ("I sink it is fun.", "I think it fun.", "I think it is fun.", False, "medium", "Missing copula"),
        ("I have free cookies.", "I've got three cookies.", "I have three cookies.", True, "hard", "I've got three"),
        ("She don't like apples.", "Doesn't she like apples?", "She doesn't like apples.", False, "adversarial", "Question changes speech act"),
        ("The ship is sailing on the see.", "See the ship sailing on the sea.", "The ship is sailing on the sea.", False, "hard", "Imperative rewrite"),
        ("He have two cats.", "Cats he have two.", "He has two cats.", False, "easy", "Still have"),
        ("We was playing tag.", "We weren't playing tag.", "We were playing tag.", False, "adversarial", "Negation flips meaning"),
        ("Yesterday I goed to the park.", "Tomorrow I will go to the park.", "Yesterday I went to the park.", False, "easy", "Time flipped"),
        ("I sink it is fun.", "I think it is not fun.", "I think it is fun.", False, "adversarial", "Polarity flip"),
        ("She don't like apples.", "She doesn't like apples at all.", "She doesn't like apples.", True, "medium", "Intensifier ok"),
        ("He have two cats.", "He has two cat.", "He has two cats.", False, "medium", "Agreement on noun lost"),
        ("The ship is sailing on the see.", "The ship is sailing on the C.", "The ship is sailing on the sea.", False, "adversarial", "Letter C"),
        ("I have free cookies.", "I have three cookie.", "I have three cookies.", False, "medium", "Plural lost"),
        ("We was playing tag.", "Me were playing tag.", "We were playing tag.", False, "medium", "Wrong subject"),
        ("Yesterday I goed to the park.", "Yesterday I went to the park with my dog.", "Yesterday I went to the park.", True, "hard", "Added dog - extra content; should reject"),
    ]
    # Fix the two I marked inconsistently:
    # "She doesn't like apple" -> False
    # "went to the park with my dog" -> False (instruction: reject totally different / extra)
    mistake_cases[42] = (
        "She don't like apples.",
        "She doesn't like apple.",
        "She doesn't like apples.",
        False,
        "hard",
        "Number change is unnecessary; treat as not the same fix quality - reject for kid judge? Actually still fixes don't. Spec: accept paraphrases that fix SAME error. Number change still fixed don't. Keep False for strict meaning.",
    )
    mistake_cases[57] = (
        "Yesterday I goed to the park.",
        "Yesterday I went to the park with my dog.",
        "Yesterday I went to the park.",
        False,
        "hard",
        "Invented extra participant",
    )

    for i, row in enumerate(mistake_cases, 1):
        spoken, child, sample, expected, diff, expl = row
        items.append(
            _judge_mistake(
                f"jm-{i:03d}",
                diff,
                spoken,
                child,
                sample,
                expected,
                expl,
                ["json", "correct boolean", "same error fixed"],
            )
        )

    banned = [
        "The cat sat on the",
        "I like to eat",
        "At night I look at the",
        "My favorite color is",
    ]
    for i in range(1, 13):
        items.append(_invent(f"inv-c-{i:03d}", "complete", banned, ""))
    for i in range(1, 13):
        items.append(
            _invent(
                f"inv-m-{i:03d}",
                "mistake",
                [],
                "Do not reuse these spoken sentences: 'She don't like apples.', 'Yesterday I goed to the park.'",
            )
        )

    # Extra context-dependent / coaching
    extra_ctx = [
        ("Bring umbrella if it will rain.", "Bring an umbrella if it rains.", ["Bring an umbrella if it rains.", "Bring an umbrella if it is going to rain."], "context_dependent", "hard", ["umbrella", "rain"]),
        ("She suggested me to wait.", "She suggested that I wait.", ["She suggested that I wait.", "She suggested waiting.", "She advised me to wait."], "word_choice", "hard", ["wait"]),
        ("I look forward to meet you.", "I look forward to meeting you.", ["I look forward to meeting you."], "verb_tense", "medium", ["meeting", "you"]),
        ("He stopped to smoke last year.", "He stopped smoking last year.", ["He stopped smoking last year.", "He quit smoking last year."], "subtle_distinctions", "adversarial", ["smoking", "year"]),
        ("I used to could swim.", "I used to be able to swim.", ["I used to be able to swim.", "I could swim."], "awkward_phrasing", "hard", ["swim"]),
        ("The police is coming.", "The police are coming.", ["The police are coming."], "subject_verb_agreement", "medium", ["police", "coming"]),
        ("Measles are a disease.", "Measles is a disease.", ["Measles is a disease."], "subject_verb_agreement", "hard", ["measles", "disease"]),
        ("Ten dollars are enough.", "Ten dollars is enough.", ["Ten dollars is enough.", "Ten dollars are enough."], "multiple_valid", "hard", ["dollars", "enough"]),
        ("I request you to kindly do the needful.", "Please do what is needed.", ["Please do what is needed.", "Please take care of this.", "Please handle this."], "natural_english", "hard", ["needed"]),
        ("Kindly revert back at the earliest.", "Please reply as soon as you can.", ["Please reply as soon as you can.", "Please reply soon."], "natural_english", "hard", ["reply"]),
        ("He is knowing to swim.", "He knows how to swim.", ["He knows how to swim.", "He can swim."], "verb_tense", "medium", ["swim"]),
        ("I am agree with you.", "I agree with you.", ["I agree with you."], "awkward_phrasing", "easy", ["agree"]),
        ("Does she has a ticket?", "Does she have a ticket?", ["Does she have a ticket?"], "subject_verb_agreement", "medium", ["ticket"]),
        ("Let's don't be late.", "Let's not be late.", ["Let's not be late.", "Let us not be late."], "awkward_phrasing", "medium", ["late"]),
        ("I didn't saw nobody.", "I didn't see anybody.", ["I didn't see anybody.", "I didn't see anyone.", "I saw nobody."], "multiple_errors", "medium", ["see"]),
        ("Whose turn is it to whose?", "Whose turn is it?", ["Whose turn is it?"], "awkward_phrasing", "adversarial", ["turn"]),
        ("The staff is arguing among themselves.", "The staff are arguing among themselves.", ["The staff are arguing among themselves.", "The staff is arguing among itself."], "multiple_valid", "adversarial", ["staff", "arguing"]),
        ("A number of students was absent.", "A number of students were absent.", ["A number of students were absent."], "subject_verb_agreement", "hard", ["students", "absent"]),
        ("The number of students were surprising.", "The number of students was surprising.", ["The number of students was surprising."], "subtle_distinctions", "adversarial", ["number", "students", "surprising"]),
        ("Hardly I had sat when the phone rang.", "Hardly had I sat when the phone rang.", ["Hardly had I sat when the phone rang.", "I had hardly sat when the phone rang."], "word_order", "hard", ["phone", "rang"]),
        ("No sooner she arrived than it rained.", "No sooner had she arrived than it rained.", ["No sooner had she arrived than it rained.", "As soon as she arrived, it rained."], "word_order", "hard", ["arrived", "rained"]),
        ("I want that he leaves.", "I want him to leave.", ["I want him to leave.", "I want that he leave."], "sentence_structure", "hard", ["leave"]),
        ("She made me to cry.", "She made me cry.", ["She made me cry."], "sentence_structure", "medium", ["cry"]),
        ("Let him to go.", "Let him go.", ["Let him go."], "sentence_structure", "easy", ["go"]),
        ("I helped him carrying the bags.", "I helped him carry the bags.", ["I helped him carry the bags.", "I helped him in carrying the bags."], "sentence_structure", "medium", ["bags"]),
        ("The man, with the hat is my uncle.", "The man with the hat is my uncle.", ["The man with the hat is my uncle."], "punctuation", "medium", ["hat", "uncle"]),
        ("My sister who is a doctor lives nearby.", "My sister, who is a doctor, lives nearby.", ["My sister, who is a doctor, lives nearby.", "My sister who is a doctor lives nearby."], "multiple_valid", "hard", ["sister", "doctor"]),
        ("Eat your vegetables they are healthy you will grow.", "Eat your vegetables; they are healthy, and you will grow.", ["Eat your vegetables. They are healthy, and you will grow.", "Eat your vegetables; they are healthy, and you will grow."], "run_on", "medium", ["vegetables", "healthy"]),
        ("Him going there was a mistake.", "His going there was a mistake.", ["His going there was a mistake.", "It was a mistake for him to go there."], "pronouns", "hard", ["going", "mistake"]),
        ("The car needs washed.", "The car needs to be washed.", ["The car needs to be washed.", "The car needs washing."], "natural_english", "hard", ["car", "washed"]),
    ]
    for i, (src, gt, acc, cat, diff, meaning) in enumerate(extra_ctx, 1):
        items.append(
            _corr(
                f"ctx-{i:03d}",
                cat,
                diff,
                src,
                gt,
                acc,
                "Context-sensitive grammar; preserve the intended proposition.",
                ["preserve meaning"],
                [],
                meaning,
            )
        )

    # Deduplicate ids
    seen = set()
    unique = []
    for it in items:
        if it["id"] in seen:
            raise SystemExit(f"duplicate id {it['id']}")
        seen.add(it["id"])
        unique.append(it)

    return {
        "version": 1,
        "task_family": "english_grammar_sentence_coaching",
        "notes": (
            "Ground truth prefers standard American school English. "
            "acceptable_answers lists linguistically valid alternatives. "
            "already_correct items must not be rewritten into a different meaning."
        ),
        "examples": unique,
    }


def main() -> None:
    data = build()
    path = HERE / "prompts.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cats: dict[str, int] = {}
    diffs: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for ex in data["examples"]:
        cats[ex["category"]] = cats.get(ex["category"], 0) + 1
        diffs[ex["difficulty"]] = diffs.get(ex["difficulty"], 0) + 1
        tasks[ex["task"]] = tasks.get(ex["task"], 0) + 1
    print(f"wrote {path} n={len(data['examples'])}")
    print("tasks", json.dumps(tasks, sort_keys=True))
    print("difficulties", json.dumps(diffs, sort_keys=True))
    print("categories", json.dumps(cats, sort_keys=True))


if __name__ == "__main__":
    main()
