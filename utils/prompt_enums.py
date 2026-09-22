from enum import Enum

class PROMPTS(Enum):
    BASE_PROMPT_EMBED = """What is the most likely word/phrase to replace "BLANK"? Respond ONLY with the word/phrase.

    {prompt}

    Option A: {a}
    Option B: {b}

    BLANK="""
    

    EXPLANATION_EMBED="""What is the most likely word/phrase to replace "BLANK"? Explain your reasoning.

    {prompt}

    Option A: {a}
    Option B: {b}

    BLANK="""

    EXPLANATION_CODE_EMBED="""
        You are a classifier. Based on the instructions embedded in the prompt text,
        output the correct category label(s) based on the explanation. Respond wih comma separated values, always ordered like so:\n
         ASSUME: 0, RECAP: 0, SIM: 0, MECH: 1, PERSPECT: 1, MAP: 0, RULE: 0, CONTRAST: 0, UNCERTAIN: 0, DEFLECT: 0\n
        no explanation or punctuation. No special characters or asterisks.\n\n
        Use the following coding schema:\n"
        | Code          | Label                                 | Description                                                                        |\n
        | ------------- | ------------------------------------- | ---------------------------------------------------------------------------------- |\n
        | **ASSUME**    | Assumption / Guess                    | Reasoning based on incomplete info, explicitly mentioning assumption or guesswork. |\n
        | **RECAP**     | Recapitulation                        | Restating or lightly paraphrasing the prompt with no added inference.              |\n
        | **SIM**       | Simulation / Embodied Imagery         | Reasoning grounded in visualizing or acting out the scene.                         |\n
        | **MECH**      | Mechanistic / Physical                | Reasoning based on forces, containment, causality, or physics.                     |\n
        | **PERSPECT**  | Agent-Centered Perspective-Taking     | Inferring from the perspective of a character (left/right, front/back, etc.).      |\n
        | **MAP**       | Fixed Frame of Reference (Geospatial) | Reasoning based on cardinal directions or map-fixed coordinates.                   |\n
        | **RULE**      | General Rule / World Knowledge        | Applying general truths or routines.                                               \n
        | **CONTRAST**  | Contrastive Elimination               | Arguing why other option(s) are invalid or less plausible.                         |\n
        | **UNCERTAIN** | Meta-Cognitive Uncertainty            | Expresses doubt, correction, or hedging.                                           |\n
        | **DEFLECT**   | Humor / Deflection                    | Non-serious or deflective reasoning (often joking or vague).                       |\n
    """


    GEMMA_IT_PROMPT_EMBED = """<start_of_turn>user
    What is the most likely word/phrase to replace "BLANK"? Respond only with 'BLANK=[word/phrase].'

    {prompt}

    Option A: {a}
    Option B: {b}<end_of_turn>user
    <start_of_turn>model
    BLANK="""

    GEMMA_IT_PROMPT_EMBED_JUSTIFY = """<start_of_turn>user
    What is the most likely word/phrase to replace "BLANK"? Think step by step, then respond with 'BLANK=[word/phrase].'

    {prompt}

    Option A: {a}
    Option B: {b}<end_of_turn>user
    <start_of_turn>model
    BLANK="""

    LLAMA_IT_PROMPT_EMBED = """<s>[INST]
    What is the most likely word/phrase to replace "BLANK"? Respond only with 'BLANK=[word/phrase].'

    {prompt}

    Option A: {a}
    Option B: {b} [/INST]
    BLANK="""
