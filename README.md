参考来源：https://github.com/wzlj/Learning/tree/master/231717
```
训练数据
image_1.png ：47161 * 50141, PNG, RGBA
image_2.png ：77470 * 46050, PNG, RGBA
测试数据
image_3.png ：37241 * 19903, PNG, RGBA
image_4.png ：25936 * 28832, PNG, RGBA
```
第一次提交：test_submit.py
直接读取大图，一边取512*512块，一边预测！
score:0.2357

第二次提交：test_submit_lq.py
先将大图切割为512 *512 的小图，然后再依次送入预测！
score:0.2455

#image_3.png
![mage](https://github.com/liuliu408/20190719/blob/master/utils/image_3_resize.png)


#image_4.png
![image](https://github.com/liuliu408/20190719/blob/master/utils/image_4_resize.png)
