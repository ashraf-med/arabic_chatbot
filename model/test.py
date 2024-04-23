from model_utils import classify_text , get_label

text1 = '? مرحبا هل يمكنني أن أسألك'
text2 =' اريد تحدث مع شخص  '
text3 = 'هل يمكنني اخد المال'
text4 = 'اريد تتبع الطلبية الخاص بي'
text5 = 'مرحبا اريد الغاء عملية الشراء  '


print(get_label(classify_text(text5)))
print(get_label(classify_text(text1)))
