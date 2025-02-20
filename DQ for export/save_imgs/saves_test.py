
a = "save_imgs/seeded-by-week-run/net_pnls.txt"
b = "save_imgs/seeded-tue-mon-run/net_pnls.txt"
c = "save_imgs/seeded-wed-tue-run/net_pnls.txt"
d = "save_imgs/seeded-thu-wed-run/net_pnls.txt"
e = "save_imgs/seeded-fri-thu-run/net_pnls.txt"

fn = d

def pa(fns):
    for fn in fns:
        netpnl = 0
        wins = 0
        wins1rp = 0
        netw = 0
        losses = 0
        netl = 0

        with open(fn) as f:
            for i in f.readlines():
                n = int(i.replace("\n", "").split(": ")[-1])
                netpnl += int(n)
                if n > 0:
                    wins += 1
                    netw += n
                    # if n > 250:
                    #     wins -= 1
                    #     netw -= n
                    #     netpnl -= int(n)
                    # elif n > 250:
                    #     wins1rp += 1

                    if n > 250:
                        wins1rp += 1
                elif n < 0:
                    losses += 1
                    netl += n
                # print()

        aaa = [str(netpnl), str(wins), str(wins1rp), str(losses), str(netw), str(netl), str(wins / (losses + wins))[:4], str(wins1rp / wins)[:4], "\n\n"]

        print(fn)
        print(", ".join(aaa))
        # print(wins, wins1rp, losses)
        # print(netw, netl, "\n\n")

# pa([b, c, d, e])

pa([a, b, c, d, e])