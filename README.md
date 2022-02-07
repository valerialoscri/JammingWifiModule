# JammingWifiModule
A jamming Wifi Module For NS-3

This is an ns-3 module that can be used to perform simulations of a WiFi network.

[Documentation](/guides/content/editing-an-existing-page) 

## Getting Started
### Prerequisites

You need to install ns-3 and clone this repositorie in src. 

``` git clone https://github.com/nsnam/ns-3-dev-git ns-3 ```

``` git clone ```

Moroever, to use and create Smart Mitigation method and Smart Attack, we need to install the ["ns-3 gym"](https://apps.nsnam.org/app/ns3-gym/) module

```git clone https://github.com/tkn-tub/ns3-gym.git```

### Compilation 

In the ns-3 folder compile and build the code

``` cd ns-3```

```./waf configure --enable-tests --enable-examples```

./waf build```

Finally, make sure tests run smoothly with:

```./test.py -s jamming```

If the script returns that the lorawan test suite passed, you are good to go. Otherwise, if tests fail or crash, consider filing an issue

### Licence

This software is licensed under the terms of the GNU GPLv2 (the same license that is used by ns-3). See the LICENSE.md file for more details.



