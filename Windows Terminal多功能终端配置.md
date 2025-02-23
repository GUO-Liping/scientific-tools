# Windows Terminal多功能终端配置

## 目的：为Windows Terminal终端配置丰富多彩的主题、实现命令自动补全（类似zsh）

## 步骤1：安装Windows Terminal

前往 [Windows Terminal - Microsoft Store 应用程序](https://link.juejin.cn/?target=https%3A%2F%2Fapps.microsoft.com%2Fstore%2Fdetail%2Fwindows-terminal%2F9N0DX20HK701%3Fhl%3Dzh-cn%26gl%3Dcn)， 点击下载/获取；

## 步骤2：安装字体

进入[Nerd Fonts - Iconic font aggregator, glyphs/icons collection, & fonts patcher](https://link.juejin.cn/?target=https%3A%2F%2Fwww.nerdfonts.com%2Ffont-downloads)，下载喜欢的字体，例如，选择：Caskaydia Cove Nerd Font。下载解压后，打开“C:\Windows\Fonts”文件夹，将解压后.ttf格式的字体粘贴进去；打开：Windows Terminal设置→配置文件→默认值→外观→选择字体（例如：Caskaydia Cove NF）

## 步骤3：安装Powershell

前往[PowerShell - Microsoft Store应用商店](https://apps.microsoft.com/detail/9mz1snwt0n5d?hl=zh-cn&gl=CN)，点击下载/获取。然后，启动Windows Terminal→默认配置文件→选择`PowerShell`(请注意，不是`Windows Powershell`)；

## 步骤4：安装oh-my-posh，posh-git和Readline

逐行输入以下命令

```
Install-Module -Name PowerShellGet -Force
winget install JanDeDobbeleer.OhMyPosh -s winget
PowerShellGet\Install-Module posh-git -Scope CurrentUser -Force
Install-Module PSReadLine
```

## 步骤5：修改配置文件-选择主题风格

打开Windows Terminal，输入：

```
notepad.exe $PROFILE
```

打开$PROFILE文件后，加入以下内容：

```
Set-PSReadLineKeyHandler -Key Tab -Function MenuComplete #Tab键会出现自动补全菜单
Set-PSReadlineKeyHandler -Key UpArrow -Function HistorySearchBackward
Set-PSReadlineKeyHandler -Key DownArrow -Function HistorySearchForward
# 上下方向键箭头，搜索历史中进行自动补全

oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH/jandedobbeleer.omp.json" | Invoke-Expression
Import-Module posh-git # git的自动补全

```

其中“jandedobbeleer”表示一种主题风格，可通过替换该字段实现Windows Terminal主题风格的改变

oh-my-posh中支持的所有主题格式可通过以下命令查看：

```
Get-PoshThemes
```

比如，可将文件中的字段"jandedobbeleer"修改为"wholespace"，修改之后保存，关闭。

最后采用以下命令：

```
. $PROFILE
```

修改便生效！

此外，--config 关键字内还支持以下两种内容形式：

--config 'C:/Users/Posh/jandedobbeleer.omp.json'

--config 'https://raw.githubusercontent.com/JanDeDobbeleer/oh-my-posh/main/themes/jandedobbeleer.omp.json'



## 参考资料

[1] [Home | Oh My Posh](https://ohmyposh.dev/)

[2] [Windows Terminal 完美配置 PowerShell 7.1 - 知乎](https://zhuanlan.zhihu.com/p/137595941)

