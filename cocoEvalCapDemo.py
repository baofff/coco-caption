
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu


def get_score(gts, res):
    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    ]

    eval = {}
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                eval[m] = sc
        else:
            eval[method] = score
    return eval


gts = {
    28377: [{'image_id': 28377, 'caption': 'A street with various buildings on each side and a clock tower.'},
            {'image_id': 28377, 'caption': 'A narrow lane has buildings on either side and one of the buildings is yellow and another is yellow and white.'},
            {'image_id': 28377, 'id': 314546, 'caption': 'There is a church at the end of the street.'},
            {'image_id': 28377, 'id': 314579, 'caption': 'An alley way of a church and buildings with balconies'},
            {'image_id': 28377, 'id': 317669, 'caption': 'A city street surrounded by tall colorful buildings.'}],
    239448: [{'id': 584162, 'caption': 'Two men standing on a very tall clock tower with a white clock and two thermometers.'},
             {'id': 584702, 'caption': 'Two cowboys statues are at the top of a tower. '},
             {'image_id': 239448, 'id': 586526, 'caption': 'Tower clock designed with two western shooters for entertainment display'},
             {'caption': 'a clock with two gunman from the old west'},
             {'image_id': 239448, 'caption': 'A clock tower with two statues of cowboys on it '}],
    558524: [{'image_id': 558524, 'id': 612929, 'caption': 'The large, very old jar is on display behind glass.'},
             {'caption': 'An intricately designed vase is shown in a glass case.'},
             {'image_id': 558524, 'caption': 'Detailed vase on display on a white pedestal. '},
             {'image_id': 558524, 'id': 613901, 'caption': 'A vase sitting on a small table as a display in a room '},
             {'image_id': 558524, 'id': 615227, 'caption': 'A brown and gold antique vase being displayed.'}],
}


res = {
    28377: [{'caption': 'large clock tower in front of a building'}],
    239448: [{'image_id': 239448, 'caption': 'clock tower with a clock on the side of a building'}],
    558524: [{'caption': 'white and white vase that is sitting on top of a table'}],
}


print(get_score(gts, res))
