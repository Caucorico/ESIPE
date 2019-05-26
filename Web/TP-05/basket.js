"use strict";

function getBasket()
{
	var ajax = new XMLHttpRequest();

	ajax.onreadystatechange = function() {
		let totalNumber = 0;

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
				totalNumber += line.quantity;
				tab.appendChild(newLine);
			});

			let quantitySpan = document.getElementById("quantity");
			quantitySpan.innerText = totalNumber;
		}
	};

	ajax.open("GET", "fruits.json", true);
	ajax.overrideMimeType("application/json");
	ajax.send();
}