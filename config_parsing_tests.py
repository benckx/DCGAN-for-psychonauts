import unittest

import yaml


class TestStringMethods(unittest.TestCase):

  def test_upper(self):
    self.assertEqual('foo'.upper(), 'FOO')

  def test_isupper(self):
    self.assertTrue('FOO'.isupper())
    self.assertFalse('Foo'.isupper())

  def test_split(self):
    s = 'hello world'
    self.assertEqual(s.split(), ['hello', 'world'])
    # check that s.split fails when the separator is not a string
    with self.assertRaises(TypeError):
      s.split(2)

  def test_parse_yaml(self):
    stream = open('test_configs/test_config01.yaml', 'r')
    docs = yaml.load(stream)
    print(docs)
    print(docs[0].items())
    for x, y in docs[0].items():
      print(x, y)


if __name__ == '__main__':
  unittest.main()
