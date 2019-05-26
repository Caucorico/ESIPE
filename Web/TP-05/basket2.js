"use strict";

function getJson( path ) {
  var ajax = new XMLHttpRequest();

  ajax.onreadystatechange = function() {
    if ( ajax.readyState == 4 ) {
      let basket = JSON.parse(ajax.responseText);

      buildFunction( basket );
    }
  };

  ajax.open("GET", path, true);
  ajax.overrideMimeType("application/json");
  ajax.send();
}

function buildTotalPrice( line, totalPrice ) {
  if ( line.name === "orange" )
  {

  }
  else if ( line.name === "banana" )
  {

  }
  else if ( line.name === "peer" )
  {

  }
}

function treatFruits( jsonBasket )
{
  let totalNumber = 0;
  let number;
  let quantitySpan = document.getElementById("quantity");
  let totalPrice = 0;

  jsonBasket.map(( line ) => {
    number = buildBasket( jsonBasket );
    totalNumber += number;
    totalPrice += buildTotalPrice()
  });

  quantitySpan.innerText = totalNumber;
}

function getTotalPrice() {
  getJson( "prices.json");
}

function buildBasket( jsonBasket ) {
  let tab = document.getElementById("basket");
  let newLine = document.createElement("tr");
  let fruit = document.createElement("td");
  let quantity = document.createElement("td");
  fruit.innerHTML = line.name;
  quantity.innerHTML = line.quantity;
  newLine.appendChild(fruit);
  newLine.appendChild(quantity);
  tab.appendChild(newLine);
}

function getBasket() {
  getJson( "fruits.json");
}