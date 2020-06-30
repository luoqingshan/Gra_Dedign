#!/bin/python

import wx
from PIL import Image
import numpy as np
import os
from tst import prediect


#文件-hello
def OnHello(event):
    wx.MessageBox("遥感图像识别")

#关于
def OnAbout(event):
    """Display an About Dialog"""
    wx.MessageBox("这是一个识别单张图像的界面-罗",
                  "关于 Hello World",
                  wx.OK | wx.ICON_INFORMATION)


class HelloFrame(wx.Frame):

    def __init__(self,*args,**kw):
        super(HelloFrame,self).__init__(*args,**kw)

        pnl = wx.Panel(self)

        self.pnl = pnl

        st = wx.StaticText(pnl, label="选择一张图像进行识别", pos=(200, 0))
        font = st.GetFont()
        font.PointSize += 10
        font = font.Bold()
        st.SetFont(font)

        # 选择图像文件按钮
        btn = wx.Button(pnl, -1, "select",pos=(4,4))
        btn.SetBackgroundColour("#0a74f7")
        #事件
        btn.Bind(wx.EVT_BUTTON, self.OnSelect)


        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("欢迎来到图像识别系统")

    #菜单栏
    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H",
                                    "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator()

        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "Help")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, OnAbout, aboutItem)
    #退出
    def OnExit(self, event):
        self.Close(True)

    #select按钮设置
    def OnSelect(self, event):
        wildcard = "image source(*.jpg)|*.jpg|" \
                   "Compile Python(*.pyc)|*.pyc|" \
                   "All file(*.*)|*.*"
        dialog = wx.FileDialog(None, "Choose a file", os.getcwd(),
                               "", wildcard, wx.ID_OPEN)
        if dialog.ShowModal() == wx.ID_OK:

            print(dialog.GetPath())
            img = Image.open(dialog.GetPath())
            imag = img.resize([128, 128])
            image = np.array(img)

            self.initimage(name= dialog.GetPath())

            #从tst.py获取 结果
            result = prediect(image)


            result_text = wx.StaticText(self.pnl, label='', pos=(600, 400), size=(150,50))
            result_text.SetLabel(result)

            font = result_text.GetFont()
            font.PointSize += 8
            result_text.SetFont(font)
            self.initimage(name= dialog.GetPath())

# 生成图片控件
    def initimage(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(0,30), size=(600,400))
        return sb


if __name__ == '__main__':

    app = wx.App()
    frm = HelloFrame(None, title='老罗的识别器', size=(1000,600))
    frm.Show()
    app.MainLoop()