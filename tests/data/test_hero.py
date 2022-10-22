from draft.data import hero


def test_loads_and_dumps():
    text = '{"id": 1, "name": "anti_mage", "localized_name": "Anti Mage"}'
    loaded_hero = hero.Hero.loads(text)
    dumped_text = loaded_hero.dumps()
    assert text == dumped_text
    assert loaded_hero._id == 1
