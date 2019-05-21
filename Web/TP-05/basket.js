"use strict";

function getBasket()
{
	var ajax = new XMLHttpRequest();

	ajax.onreadystatechange = function() {
		if ( ajax.readyState == 4 )
		{
			let basket = JSON.parse(ajax.responseText);
	  		console.log(basket);

	  		let tab = document.getElementById("basket");

			basket.map(( line ) => {
				let newLine = document.createElement("tr");
				let fruit = document.createElement("td");
				let quantity = document.createElement("td");
				fruit.innerHTML = line.name;
				quantity.innerHTML = line.quantity;
				newLine.appendChild(fruit);
				newLine.appendChild(quantity);
				tab.appendChild(newLine);
			});
		}
	};

	ajax.open("GET", "fruits.json", true);
	ajax.overrideMimeType("application/json");
	ajax.send();
}