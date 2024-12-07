from policy import Policy

class Policy2210xxx(Policy):
    def __init__(self):
        # Student code here
        self.sorted_prods = []
        self.sorted = False

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stock_idx = -1

        if not self.sorted:
            for prod in list_prods:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                prod_area = prod_w * prod_h
                self.sorted_prods.append((prod_area, prod))

            # Sort products by area in descending order
            self.sorted_prods.sort(key=lambda x: x[0], reverse=True)
            self.sorted = True  # Đánh dấu là đã sắp xếp

        for prod in self.sorted_prods:
            prod_width, prod_height = prod[1]["size"]
            placed = False
            x, y = 0, 0  # Initialize x and y
            if prod[1]["quantity"] > 0:
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    for (w, h) in [(prod_width, prod_height), (prod_height, prod_width)]:
                        for x in range(stock_w - w + 1):
                            for y in range(stock_h - h + 1):
                                if self._can_place_(stock, (x, y), (w, h)):
                                    stock_idx = i
                                    placed = True
                                    print("Product size: ", (w, h), "Quantity: ", prod[1]["quantity"])
                                    return {"stock_idx": stock_idx, "size": (w, h), "position": (x, y)}
                            if placed:
                                break
                        if placed:
                            break
                    if placed:
                        break

        return {"stock_idx": stock_idx, "size": prod[1]["size"], "position": (x, y)}